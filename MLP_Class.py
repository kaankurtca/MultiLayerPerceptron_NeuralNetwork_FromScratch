import numpy as np

class MLPerceptron():

    def __init__(self, inputDim=100, hiddenLayers = [3, 5], outputDim=4):
        self.inputDim = inputDim
        self.hiddenLayers = hiddenLayers
        self.outputDim = outputDim

        layers= [self.inputDim] + hiddenLayers + [outputDim]

        weights=[]
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

    def feedForward(self, inputs):
        activations = inputs

        for w in self.weights:
            # calculate matrix multiplication between previous activation and weight matrix
            v = np.dot(activations, w)
            activations = self.sigmoid(v)
        return activations

    def sigmoid(self, x):

        y = 1.0 / (1 + np.exp(-x))
        return y


if __name__ == '__main__':
    mlp=MLPerceptron()
