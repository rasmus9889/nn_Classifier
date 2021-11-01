# Spam Classifier Neural Net
Uses a custom made neural net and available data to train a model for classifying spam in the 'data' folder
Implemented utilizing a single output neuron, which outputs a value between 0 and 1, which is then rounded to output either 1 for True(ham) and 0 for False(spam).
The code is structured utilizing 'layer' objects, which each contain an amount of neurons, their biases, and their corresponding incoming weights.
