import gzip
import sys
import time
import numpy as np
from sklearn.utils import shuffle


def get_data(sample_size, pathX, pathY):
    f = gzip.open(pathX, 'r')

    # read off unimportant bytes describing file protocol
    image_size = 28 * 28
    protocol_length = 16
    f.read(protocol_length)

    X = f.read(image_size * sample_size)
    X = np.frombuffer(X, dtype=np.uint8).astype(np.float32)
    X = X.reshape(sample_size, image_size)

    f = gzip.open(pathY, 'r')

    protocol_length = 8
    f.read(protocol_length)
    Y_temp = f.read(sample_size)
    Y_temp = np.frombuffer(Y_temp, dtype=np.uint8)

    Y = np.zeros([sample_size, 10], dtype='f')
    for sample in range(sample_size):
        Y[sample][Y_temp[sample]] = 1.0

    return [np.array([data / 255.0 for data in X], dtype='f'), Y]

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.w = []
        self.b = []
        for layer in range(1, len(layers)):
            self.w.append((2.0 * (np.random.randint(1e9, size=[layers[layer], layers[layer - 1]]) / 1e9)) - 1.0)
            self.b.append(((2.0 * (np.random.randint(1e9, size=[layers[layer]]) / 1e9)) - 1.0).reshape([layers[layer], 1]))

    def sigmoid(self, data):
        return np.where(data >= 0, 1 / (1 + np.exp(-data)), np.exp(data) / (1 + np.exp(data)))

    def sigmoid_derivative(self, data):
        temp = self.sigmoid(data)
        return temp * (1 - temp)

    def backPropFast(self, train_X, train_Y, eta):
        m = len(train_X[0])

        gradients_w = [np.zeros([self.layers[layer], self.layers[layer - 1]], dtype='f') for layer in range(1, len(self.layers))]
        gradients_b = [np.zeros([self.layers[layer]], dtype='f') for layer in range(1, len(self.layers))]
        activations = [train_X]
        zs = [train_X]

        for layer in range(1, len(self.layers)):
            zs.append(np.dot(self.w[layer - 1], activations[-1]) + self.b[layer - 1])
            activations.append(self.sigmoid(zs[-1]))

        delta_l = ((activations[-1] - train_Y) * self.sigmoid_derivative(zs[-1]))
        gradients_b[-1] = delta_l.sum(axis=1)
        gradients_w[-1] = np.dot(delta_l, np.transpose(activations[-2]))

        for layer in range(len(self.layers) - 2, 0, -1):
            delta_l = np.dot(np.transpose(self.w[layer]), delta_l) * self.sigmoid_derivative(zs[layer])
            gradients_b[layer - 1] += delta_l.sum(axis=1)
            gradients_w[layer - 1] += np.dot(delta_l, np.transpose(activations[layer - 1]))
        
        for layer in range(len(gradients_b)):
            self.w[layer] -= eta * gradients_w[layer] / float(m)
            self.b[layer] -= eta * gradients_b[layer].reshape([len(gradients_b[layer]), 1]) / float(m)

    def train(self, train_X, train_Y, eta, batch_size):
        m = len(train_X[0]) #60000
        for batch in range(int(m / batch_size)):
            X = train_X[ : , batch * batch_size : (batch + 1) * batch_size]
            Y = train_Y[ : , batch * batch_size : (batch + 1) * batch_size]
            self.backPropFast(X, Y, eta)

    def getAccuracy(self, test_X, test_Y):
        for layer in range(len(self.layers) - 1):
            test_X = self.sigmoid(np.dot(self.w[layer], test_X) + self.b[layer].reshape([self.layers[layer + 1], 1]))
        predictions = np.argmax(test_X, 0)
        Y = np.argmax(test_Y, 0)
        print("Accuracy:", str(np.sum(predictions == Y) * 100.0 / float(len(predictions))) + "%")

if __name__ == "__main__":
    train_size = 60000
    test_size = 10000

    """ python attempt_3.py [784, 30, 10] 10 3.0 30 """
    layers = [int(i) for i in sys.argv[1].strip('][').split(',')]   # [784, 30, 10]
    batch_size = int(sys.argv[2])                                   # 10
    eta = float(sys.argv[3])                                        # 3.0
    epochs = int(sys.argv[4])                                       # 10

    train_X, train_Y = get_data(train_size, 'MNIST_Dataset/train-images-idx3-ubyte.gz', 'MNIST_Dataset/train-labels-idx1-ubyte.gz')
    test_X, test_Y = [np.transpose(a) for a in get_data(test_size, 'MNIST_Dataset/t10k-images-idx3-ubyte.gz', 'MNIST_Dataset/t10k-labels-idx1-ubyte.gz')]

    network = Network(layers)
    for epoch in range(epochs):
        train_X, train_Y = shuffle(train_X, train_Y)
        start_time = time.perf_counter()
        print("\nEpoch:", epoch + 1)
        network.train(np.transpose(train_X), np.transpose(train_Y), eta, batch_size)
        network.getAccuracy(test_X, test_Y)
        end_time = time.perf_counter()
        print(f"Time taken in this epoch = {end_time - start_time:0.4f} seconds")
