'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn.preprocessing import label_binarize
from PIL import Image
import copy

class NeuralNet:

    def __init__(self, layers, epsilon=0.2, learningRate=2, numEpochs=100):
        '''
        Constructor
        Arguments:
            layers - a numpy array of L-2 integers (L is # layers in the network)
            epsilon - one half the interval around zero for setting the initial weights
            learningRate - the learning rate for backpropagation
            numEpochs - the number of epochs to run during training
        '''

        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
          
        self.lamda = 0.001
        self.all_layers_info = None
        self.L = 0
        self.theta = dict()
        self.a_Val = dict()
        self.delta = dict()
        self.gradient = dict()
        self.count = 0

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n, d = X.shape
        # transform y into an n-by-10 numpy array (unique_y = 10)
        num_unique_y = len(np.unique(y))
        binary_y = label_binarize(y, classes = np.unique(y))

        self.all_layers_info = np.append(np.append(d, self.layers), num_unique_y)
        # print self.all_layers_info
        self.L = len(self.all_layers_info)

        np.random.seed(28)
        # Initialize theta
        for l in range(self.L - 1):
            self.theta[l + 1] = np.random.uniform(low=-self.epsilon, high=self.epsilon, size=(self.all_layers_info[l + 1], (self.all_layers_info[l] + 1)))
            # print self.theta[l+1][0]
        # loop though Epochs
        for i in range(self.numEpochs):
            self._forwardPropagation_(X)
            self._backPropagation_(binary_y)


    def _forwardPropagation_(self, X):
        '''
        Calculate nodes of next layer
        Arguments:
            x is a 1-by-d numpy array
            theta is the weight matrix
        Functions:
            updates all layers
        '''
        n,d = X.shape

        self.a_Val[0] = np.c_[np.ones(n), X]

        for i in range(self.L - 1):
            temp_a = self._sigmoid_(self.a_Val[i].dot(self.theta[i + 1].T))
            temp_a = np.c_[np.zeros(n), temp_a]
            self.a_Val[i + 1] = temp_a

        self.a_Val[self.L - 1] = self.a_Val[self.L - 1][:, 1:]

    def _backPropagation_(self, y):
        '''
        Update error matrix
        '''
        self.delta[self.L - 1] = self.a_Val[self.L - 1] - y

        for l in reversed(range(0, self.L - 1)):
            temp_1 = self.delta[l + 1].dot(self.theta[l + 1])[:, 1:]
            temp_2 = np.multiply(self.a_Val[l][:, 1:], 1 - self.a_Val[l][:, 1:])
            self.delta[l] = np.multiply(temp_1, temp_2)

            self.gradient[l + 1] = self.delta[l + 1].T.dot(self.a_Val[l]) / len(y)

            temp_t = self.theta[l + 1][:, 1:]
            d = temp_t.shape[0]
            temp_t = np.concatenate((np.zeros((d, 1)), temp_t), axis = 1)
            self.gradient[l + 1] = self.gradient[l + 1] + temp_t * self.lamda
            
            self.theta[l + 1] = self.theta[l + 1] - self.learningRate * self.gradient[l + 1]


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        self.visualizeHiddenNodes("./hidden_layer.png")
        self._forwardPropagation_(X)
        pred_y = np.argmax(self.a_Val[self.L - 1], axis=1)
        return pred_y
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        # self.theta[1].shape = (25, 401)

        hidden_layer = dict()

        if self._isSquare_(self.theta[1].shape[0]):
            theta_without_bias = copy.deepcopy(self.theta[1][:, 1:])   # (25, 400)

            d, n = theta_without_bias.shape

            # d = 25
            # n = 400

            for i in range(d):
                temp_max = np.max(theta_without_bias[i, :])
                temp_min = np.min(theta_without_bias[i, :])
                # print theta_without_bias[i]
                theta_without_bias[i] = 255 * (theta_without_bias[i] - temp_min) / (temp_max - temp_min)
                hidden_layer[i] = theta_without_bias[i].reshape(np.sqrt(n), np.sqrt(n))
                # self.hidden_layer[25][20, 20]

            img = Image.new('L', (int(np.sqrt(d*n)), int(np.sqrt(d*n))), 'black')
            pixel = img.load()

            row_index = 0
            row_position = 0
            column_index = 0
            column_position = 0
            for i in range(img.size[0]):
                row_index = i/ int(np.sqrt(n))
                row_position = i % int(np.sqrt(n))
                for j in range(img.size[1]):
                    column_index = j / int(np.sqrt(n))
                    column_position = j % int(np.sqrt(n))
                    column_position = j % int(np.sqrt(n))
                    pixel[i,j] = int(hidden_layer[int(np.sqrt(d)) * row_index + column_index][row_position, column_position])

            img.show()
            img.save(filename, 'PNG')

        else:
            return 0

    def _sigmoid_(self, z):
        '''
        z has to be an array in numpy
        '''
        return 1./(1 + np.exp(-z))

    def _isSquare_(self, z):
        temp = np.sqrt(int(z))
        if "." in str(abs(int(temp))):
            return False
        else:
            return True
