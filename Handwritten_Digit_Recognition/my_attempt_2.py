import gzip
import numpy as np
import matplotlib.pyplot as plt
import timeit
from random import random
import sys

def get_data(sample_size):
    """
    Returns:
    np.ndarray: (normalized array of flattened images, float, array of labels, int)
    """

    file_path = 'train-images-idx3-ubyte.gz'
    f = gzip.open(file_path, 'r')

    # read off unimportant bytes describing file protocol
    protocol_length = 16
    f.read(protocol_length)

    image_size = 28 * 28

    train_data = f.read(image_size * sample_size)
    train_data = np.frombuffer(train_data, dtype=np.uint8).astype(np.float32)
    train_data = train_data.reshape(sample_size, image_size)

    file_path = 'train-labels-idx1-ubyte.gz'
    f = gzip.open(file_path, 'r')

    protocol_length = 8
    f.read(protocol_length)
    train_lables = f.read(sample_size)
    train_lables = np.frombuffer(train_lables, dtype=np.uint8)

    return [np.array([data / 255.0 for data in train_data], dtype='f'), train_lables]

def show_image(img):
    if len(img) != 28 * 28:
        print("image length inaccurate for printing (" + len(img) + ")")
        print("image length should be 784")
        return

    plt.imshow(img.reshape(28, 28))
    plt.show()

def sigmoid(data):
    return 1 / (1 + np.exp(-data))

def sigmoid_derivative(data):
    temp = np.exp(-data)
    return temp / ((1 + temp) ** 2)

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.w = []
        self.b = []
        for layer in range(1, len(layers)):
            self.w.append((2.0 * (np.random.randint(1e9, size=[layers[layer], layers[layer - 1]]) / 1e9)) - 1.0)
            self.b.append((2.0 * (np.random.randint(1e9, size=[layers[layer]]) / 1e9)) - 1.0)

    def backProp(self, train_X, train_Y, eta):
        m = len(train_X)
        gradients_w = [np.zeros([self.layers[layer], self.layers[layer - 1]], dtype='f') for layer in range(1, len(self.layers))]
        gradients_b = [np.zeros([self.layers[layer]], dtype='f') for layer in range(1, len(self.layers))]

        for sample in range(m):
            activations = [train_X[sample]]
            zs = [train_X[sample]]
            for layer in range(1, len(self.layers)):
                zs.append(np.dot(self.w[layer - 1], activations[-1]) + self.b[layer - 1])
                activations.append(sigmoid(zs[-1]))

            y = np.zeros([self.layers[-1]], dtype='f')
            y[train_Y[sample]] = 1.0
            delta_l = (activations[-1] - y) * sigmoid_derivative(zs[-1])

            gradients_b[-1] += delta_l
            gradients_w[-1] += np.dot(delta_l.reshape([len(delta_l), 1]), activations[-2].reshape([1, len(activations[-2])]))

            for layer in range(len(self.layers) - 2, 0, -1):
                delta_l = np.dot(np.transpose(self.w[layer]), delta_l) * sigmoid_derivative(zs[layer])
                gradients_b[layer - 1] += delta_l
                gradients_w[layer - 1] += np.dot(delta_l.reshape([len(delta_l), 1]), activations[layer - 1].reshape([1, len(activations[layer - 1])]))
        
        for layer in range(len(gradients_b)):
            self.w[layer] -= eta * gradients_w[layer] / float(m)
            self.b[layer] -= eta * gradients_b[layer] / float(m)

    def train(self, data, eta, batch_size):
        n = len(data[0])
        for batch in range(int(n / batch_size)):
            train_X = data[0][batch * batch_size : (batch + 1) * batch_size]
            train_Y = data[1][batch * batch_size : (batch + 1) * batch_size]
            self.backProp(train_X, train_Y, eta)

    def getAccuracy(self, test_X, test_Y):

        m = len(test_X)
        correct_predictions = 0
        predictions = np.zeros([self.layers[-1]], dtype=int)
        
        for sample in range(m):
            a = test_X[sample]
            for layer in range(len(self.layers) - 1):
                a = sigmoid(np.dot(self.w[layer], a) + self.b[layer])

            maxid = 0
            maxip = -1000
            for digit in range(len(a)):
                if a[digit] >= maxip:
                    maxid = digit
                    maxip = a[digit]
            predictions[maxid] += 1
            if maxid == test_Y[sample]:
                correct_predictions += 1

        print("Accuracy:", str(correct_predictions * 100 / m) + "%\n")


        

if __name__ == "__main__":
    start = timeit.default_timer()

    sample_size = 60000
    train_size = 50000
    batch_size = 100
    test_size = sample_size - train_size
    eta = 0.1
    epochs = 100


    data = get_data(sample_size)
    
    # each element in train_data is a 1D array of:
    # 784 (28 * 28) floats [0 - 1]
    # each element in train_label is an integer [0 - 9]
    train_X = data[0][:train_size]
    train_Y = data[1][:train_size]
    test_X = data[0][-test_size:]
    test_Y = data[1][-test_size:]

    network = Network([28 * 28, 16, 10])
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        network.train([train_X, train_Y], eta, batch_size)
        network.getAccuracy(test_X, test_Y)