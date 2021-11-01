import numpy as np
import random
import copy as copy
# import cupy as cp - was used during development
import numpy as cp
from pprint import pprint
import io as io

import pickle as pickle

struct = [1, 2, 4, 8, 16]
totalEpochs = 1000
# boolean that determines whether the model should train or load, set to load the optimalModel file here
boolLoad = True


class SpamClassifier:
    def __init__(self, k):
        array = np.loadtxt('data/training_spam.csv', delimiter=',')
        self.array_answers = array[:, 0]
        self.array_inputs = array[:, 1:]
        structure = struct
        self.optimalModel = layer(54, structure)
        self.optimalModel.initializeLayers()
        self.dataPoints = list(range(1000))
        random.shuffle(self.dataPoints)
        self.k = k

    def train(self):
        # defining base learning rates, which are lowered if the model does not decrease at all after 4 epochs
        learnRateWeights = 0.09
        learnRateBiases = 0.11
        # a starter minimum loss number, should always be replaced in the first epoch
        minLoss = 10000
        # counts how long since the lowest loss was achieved
        lossCounter = 0
        if boolLoad:
            self.optimalModel = load()
        else:
            for i in range(totalEpochs):
                chosen = self.dataPoints
                # training
                for ii in chosen:
                    self.optimalModel.clearInputs()
                    # give the model the needed information
                    self.optimalModel.giveInput(self.array_inputs[ii])
                    self.optimalModel.giveTarget(self.array_answers[ii])
                    self.optimalModel.updateRates(learnRateWeights, learnRateBiases)
                    # predict model output for that input, and overall loss
                    self.optimalModel.predict()
                    self.optimalModel.calculateLoss()
                    # update the model based on previous info
                    self.optimalModel.createDeltas()
                    self.optimalModel.updateWeights()
                    self.optimalModel.updateBiases()
                correctTemp = 0
                loss = 0
                testing = self.dataPoints
                # check against training data, to keep an eye out for changes in loss/accuracy for each epoch
                for ii in testing:
                    self.optimalModel.clearInputs()
                    self.optimalModel.giveInput(self.array_inputs[ii])
                    self.optimalModel.giveTarget(self.array_answers[ii])
                    self.optimalModel.predict()
                    if (self.optimalModel.predict() > 0.5 and self.array_answers[ii] == 1) or (
                            self.optimalModel.predict() < 0.5 and self.array_answers[ii] == 0):
                        correctTemp = correctTemp + 1
                    loss += self.optimalModel.calculateLoss()
                if loss < minLoss:
                    minLoss = loss
                    lossCounter = 0
                else:
                    lossCounter += 1
                if lossCounter > 4:
                    learnRateWeights = learnRateWeights * 0.9
                    learnRateBiases = learnRateBiases * 0.9
                    print("NEW LEARNING RATES APPLIED: bias:", learnRateBiases, "and weights:", learnRateWeights)
                    self.optimalModel.updateRates(learnRateWeights, learnRateBiases)
                    lossCounter = 0
                print('epoch:', i)
                print('accuracy', correctTemp / 1000)
                print('loss:', loss)
                random.shuffle(self.dataPoints)
            # save the model after the epochs(overwrites existing optimal model)
            self.optimalModel.save()

    def predict(self, data):
        predictions = []
        for i in data:
            self.optimalModel.clearInputs()
            self.optimalModel.giveInput(i)
            guess = self.optimalModel.idealizedPrediction()
            predictions.append(guess)
        return np.array(predictions)


class layer:
    def __init__(self, inputNodes, localNodes, depth=0):
        self.depth = depth
        self.inputNo = inputNodes
        self.nodeNo = localNodes.pop()
        self.nextLengths = localNodes
        self.weights = (np.random.random([self.inputNo, self.nodeNo]) - 0.5) * 4
        self.biases = np.random.randint(-2, 2, [self.nodeNo])
        self.inputs = np.zeros([self.inputNo])
        self.deltas = np.zeros([self.nodeNo])
        self.nextLayer = None
        self.valuesBeforeSigmoid = []
        self.target = -1

    # creates the all the layers in the network, using the structure variable, the first unit has to be 1,
    # since that stops the recursion
    def initializeLayers(self):
        if self.nodeNo == 1:
            pass
        else:
            self.nextLayer = layer(self.nodeNo, self.nextLengths, self.depth + 1)
            self.nextLayer.setPrevious(self)
            self.nextLayer.initializeLayers()

    # hands the input to the model, but only at the base layer, predict()
    # iterates through the model until it returns the final nodes output
    def giveInput(self, inputs):
        self.inputs = inputs

    # earlier iterations of network creation were having issues where the inputs did not clear in the whole mode,
    # this iterates through and clears all of them
    def clearInputs(self):
        self.inputs = np.zeros([self.inputNo])
        if self.nextLayer is not None:
            self.nextLayer.clearInputs()

    # hands the target to the whole network, only for training the model
    def giveTarget(self, target):
        if self.nodeNo == 1:
            self.target = target
        else:
            self.target = target
            self.nextLayer.giveTarget(target)

    # returns the output of the active layer
    def getOutput(self):
        assert (hasattr(self, 'inputs'))
        if self.nodeNo == 1:
            val = cp.dot(self.weights[:, 0], self.inputs) + self.biases[0]
            self.valuesBeforeSigmoid = [val]
            return self.sigmoid(val)
        else:
            output = []
            for i in range(self.nodeNo):
                val = cp.dot(self.weights[:, i], self.inputs) + self.biases[i]
                output.append(val)
            # creates a function that can be mapped onto the list
            func = lambda x: self.sigmoid(x)
            self.valuesBeforeSigmoid = output
            return np.array(list(map(func, output)))

    # returns the sigmoid of the value, so the output of a neuron is always between 0 and 1
    def sigmoid(self, x):
        x = np.clip(x, -100000, 100000)
        return (1 / (1 + np.exp(-x)))

    # sets the previous layer, which is useful for iterating backwards
    def setPrevious(self, layer):
        self.previous = layer
        return layer

    # runs through the structure until it returns the output of the final node, by handing outputs as inputs to each layer
    def predict(self):
        if self.nodeNo == 1:
            return self.getOutput()
        else:
            self.nextLayer.giveInput(self.getOutput())
            self.nextLayer.getOutput()
            return self.nextLayer.predict()

    # returns a one or a two for a real prediction, as opposed to the odds
    def idealizedPrediction(self):
        if self.nodeNo == 1:
            return round(self.getOutput())
        else:
            self.nextLayer.giveInput(self.getOutput())
            self.nextLayer.getOutput()
            return round(self.nextLayer.predict())

    # propagates through the layers until it calculates the total loss on the final output
    def calculateLoss(self):
        if self.nodeNo == 1:
            return self.calculateTotalLoss(self.getOutput(), self.target)
        else:
            return self.nextLayer.calculateLoss()

    # only used when there is a single node, calculates difference between prediction and target,
    # for 'punishing' the model if its too wrong
    def calculateTotalLoss(self, var, target):
        return (target - var) ** 2

    # used when learning rates have to be fine tuned in later epochs
    def updateRates(self, weightRate, biasRate):
        if self.nodeNo != 1:
            self.weightRate = weightRate
            self.biasRate = biasRate
            self.nextLayer.updateRates(weightRate, biasRate)
        else:
            self.weightRate = weightRate
            self.biasRate = biasRate

    # using the deltas variable, this changes all the weights in the right direction
    def updateWeights(self):
        if self.nextLayer is not None:
            for i in range(self.nodeNo):
                for ii in range(self.inputNo):
                    self.weights[ii, i] = self.weights[ii, i] + (self.weightRate * self.deltas[i] * self.inputs[ii])
            self.nextLayer.updateWeights()
        else:
            for i in range(self.nodeNo):
                for ii in range(self.inputNo):
                    self.weights[ii, i] = self.weights[ii, i] + (self.weightRate * self.deltas[i] * self.inputs[ii])

    # also uses delta variable for same purpose but on the biases in each node
    def updateBiases(self):
        if self.nextLayer is not None:
            for i in range(self.nodeNo):
                self.biases[i] = self.biases[i] + (self.deltas[i] * self.biasRate)
            self.nextLayer.updateWeights()
        else:
            for i in range(self.nodeNo):
                self.biases[i] = self.biases[i] + (self.deltas[i] * self.biasRate)

    # function that goes to the final node to create the deltas via createDeltas2() backwards from the node
    def createDeltas(self):
        assert (self.target != -1)
        if self.nodeNo == 1:
            self.createDeltas2()
        else:
            self.nextLayer.createDeltas()

    # this function creates the deltas(the magnitude and direction in which each node should change in response to how far off it is from the answer) by moving backward
    def createDeltas2(self):
        if hasattr(self, 'previous'):
            if self.nodeNo == 1:
                self.deltas = [(self.target - self.getOutput()) * self.transferDerivative(
                    self.sigmoid(self.valuesBeforeSigmoid[0]))]
            elif self.nodeNo != 1:
                errors = []
                for i in range(self.nodeNo):
                    error = 0.0
                    for ii in range(self.nextLayer.nodeNo):
                        error += self.nextLayer.weights[i, ii] * self.nextLayer.deltas[ii]
                    errors.append(error)
                for i in range(self.nodeNo):
                    self.deltas[i] = errors[i] * self.transferDerivative(self.sigmoid(self.valuesBeforeSigmoid[i]))
            return self.previous.createDeltas2()
        else:
            errors = []
            for i in range(self.nodeNo):
                error = 0.0
                for ii in range(self.nextLayer.nodeNo):
                    error += self.nextLayer.weights[i, ii] * self.nextLayer.deltas[ii]
                errors.append(error)
            for i in range(self.nodeNo):
                self.deltas[i] = errors[i] * self.transferDerivative(self.sigmoid(self.valuesBeforeSigmoid[i]))

    # sigmoid derivative
    def transferDerivative(self, value):
        return value * (1 - value)

    # saves the active network of layers as a model
    def save(self):
        path = 'optimalModel'
        toSave = copy.deepcopy(self)
        optimal = open(path, 'wb')
        pickle.dump(toSave, optimal)
        optimal.close()


# returns the existing model in the system
def load():
    path = 'optimalModel'
    optimal = open(path, "rb")
    toReturn = pickle.load(optimal)
    optimal.close()
    return toReturn


def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train()
    return classifier


classifier = create_classifier()

SKIP_TESTS = False

if not SKIP_TESTS:
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")