import gzip
import sys
import time

import numpy as np
import matplotlib.pyplot as plt


def get_data(sample_size, pathX, pathY):
    """
    Returns:
    List of two np.ndarrays
    [normalized array of flattened images: float,
    array of labels: int]
    """

    f = gzip.open(pathX, 'r')

    # read off unimportant bytes describing file protocol
    protocol_length = 16
    f.read(protocol_length)

    image_size = 28 * 28

    train_data = f.read(image_size * sample_size)
    train_data = np.frombuffer(train_data, dtype=np.uint8).astype(np.float32)
    train_data = train_data.reshape(sample_size, image_size)

    f = gzip.open(pathY, 'r')

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


class Network:
    def __init__(self, layers):
        self.layers = layers
        self.w = []
        self.b = []
        for layer in range(1, len(layers)):
            self.w.append((2.0 * (np.random.randint(1e9, size=[layers[layer], layers[layer - 1]]) / 1e9)) - 1.0)
            self.b.append((2.0 * (np.random.randint(1e9, size=[layers[layer]]) / 1e9)) - 1.0)

    def sigmoid(self, data):
        ## The first two lines are for suppressing overflow warnings
        data = np.where(data > 500.0, 500.0, data)
        data = np.where(data < -500.0, -500.0, data)
        return np.where(data >= 0, 
                        1 / (1 + np.exp(-data)), 
                        np.exp(data) / (1 + np.exp(data)))

    def sigmoid_derivative(self, data):
        temp = self.sigmoid(data)
        return temp * (1 - temp)

    def backProp(self, train_X, train_Y, eta):
        m = len(train_X)
        gradients_w = [np.zeros([self.layers[layer], self.layers[layer - 1]], dtype='f') for layer in range(1, len(self.layers))]
        gradients_b = [np.zeros([self.layers[layer]], dtype='f') for layer in range(1, len(self.layers))]

        for sample in range(m):
            activations = [train_X[sample]]
            zs = [train_X[sample]]
            for layer in range(1, len(self.layers)):
                zs.append(np.dot(self.w[layer - 1], activations[-1]) + self.b[layer - 1])
                activations.append(self.sigmoid(zs[-1]))

            y = np.zeros([self.layers[-1]], dtype='f')
            y[train_Y[sample]] = 1.0
            delta_l = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])

            gradients_b[-1] += delta_l
            gradients_w[-1] += np.dot(delta_l.reshape([len(delta_l), 1]), activations[-2].reshape([1, len(activations[-2])]))

            for layer in range(len(self.layers) - 2, 0, -1):
                delta_l = np.dot(np.transpose(self.w[layer]), delta_l) * self.sigmoid_derivative(zs[layer])
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
        test_X = np.transpose(test_X)
        for layer in range(len(self.layers) - 1):
            test_X = self.sigmoid(np.dot(self.w[layer], test_X) + self.b[layer].reshape([self.layers[layer + 1], 1]))
        predictions = np.argmax(test_X, 0)
        print("Accuracy:", str(np.sum(predictions == test_Y) * 100.0 / float(len(predictions))) + "%")


def shuffle(data):
    m = len(data[0])
    ids = np.arange(0, m)
    np.random.shuffle(ids)
    temp_data = [np.ndarray([m, 784], dtype='f'), np.ndarray([m], dtype=np.uint8)]
    for i in range(m):
        temp_data[0][i] = data[0][ids[i]]
        temp_data[1][i] = data[1][ids[i]]
    return temp_data


if __name__ == "__main__":

    train_size = 60000
    test_size = 10000

    n = len(sys.argv)
    assert n == 5, "Incorrect system arguments provided. Ending script..."
        
    # layers = [784, 30, 10]
    # batch_size = 10
    # eta = 5.0
    # epochs = 100
    layers = [int(i) for i in sys.argv[1].strip('][').split(',')]
    batch_size = int(sys.argv[2])
    eta = float(sys.argv[3])
    epochs = int(sys.argv[4])

    train_X, train_Y = get_data(train_size, 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    test_X, test_Y = get_data(test_size, 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

    network = Network(layers)
    for epoch in range(epochs):
        train_X, train_Y = shuffle([train_X, train_Y])

        start_time = time.perf_counter()
        print("\nEpoch:", epoch + 1)
        network.train([train_X, train_Y], eta, batch_size)
        network.getAccuracy(test_X, test_Y)
        end_time = time.perf_counter()
        print(f"Time taken in this epoch = {end_time - start_time:0.4f} seconds")