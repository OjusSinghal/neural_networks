import gzip
import numpy as np
import matplotlib.pyplot as plt
import timeit
from random import random
import sys

def get_data(sample_size):
    """
    Returns:
    List: (list of flattened images, list of labels)
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

    return train_data, train_lables

def show_image(img):
    if len(img) != 28 * 28:
        print("image length inaccurate for printing (" + len(img) + ")")
        print("image length should be 784")
        return

    plt.imshow(img.reshape(28, 28))
    plt.show()

def sigmoid(data):
    temp = np.zeros(len(data)).reshape(len(data), 1)

    for i in range(len(data)):
        x = data[i][0]
        if x > 0:
            temp1 = np.exp(-x)
            temp[i][0] = 1 / (1 + temp1)
        else:
            temp1 = np.exp(x)
            temp[i][0] = temp1 / (1 + temp1)

    return temp

def sigmoid_derivative(x):
    if x > 0: temp1 = np.exp(-x)
    else: temp1 = np.exp(x)
    return temp1 / ((1 + temp1) ** 2)

class Network:
    def __init__(self, layers):
        self.weights = []
        self.biases = []
        for layer in range(1, len(layers)):
            self.weigts.append(np.zeros(shape))

    def train(self, data, epochs, eta, batch_size):
        pass

    def predict(self, data):
        pass

    def getAccuracy(self, testX, testY):
        pass

        

if __name__ == "__main__":
    start = timeit.default_timer()

    # sample_size = 60000
    # train_size = 50000
    # batch_size = 100
    sample_size = 100
    train_size = 90
    batch_size = 10

    test_size = sample_size - train_size
    batches = int(train_size / batch_size)

    data = get_data(sample_size)
    
    # each element in train_data is a 1D array of:
    # 784 (28 * 28) integers [0 - 255]
    # each element in train_label is an integer(0 - 9)
    train_X = data[0][:train_size]
    train_Y = data[1][:train_size]
    test_X = data[0][-test_size:]
    test_Y = data[1][-test_size:]

    network = Network([28 * 28, 16, 10])
    network.train([train_X, train_Y], 1, 0.1, 10)
    accuracy = network.getAccuracy(test_X, test_Y)