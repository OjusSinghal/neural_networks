import gzip
import numpy as np
import matplotlib.pyplot as plt
import timeit
from random import random
import sys


def get_data(sample_size):
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
    # temp = np.zeros(len(data)).reshape(len(data), 1)
    # print(data.shape)
    # exit()
    # for i in range(len(data)):
    #     x = data[i][0]
    if x > 0: temp1 = np.exp(-x)
    else: temp1 = np.exp(x)
    return temp1 / ((1 + temp1) ** 2)


class Layer():
    def __init__(self, n, prev):
        self.n = n
        self.values = np.zeros(n).reshape(n, 1)
        self.z = np.zeros(n).reshape(n, 1)

        self.biases = np.zeros(n).reshape(n, 1)
        self.weights = np.zeros(n * prev).reshape(n, prev)
        # self.biases = np.random.rand(n, 1) * 2 - 1
        # self.weights = np.random.rand(n, prev) * 2 - 1


    def set_value(self, pre_layer):
        self.z = np.dot(self.weights, pre_layer) + self.biases
        self.values = sigmoid(self.z)

    def get_value(self):
        return self.values

    def get_z(self, i):
        return self.z[i]

    def get_weights(self, i):
        return self.weights[i]

    def update_weights(self, eta, derivative):
        self.weights -= eta * derivative

    def update_biases(self, eta, derivative):
        self.biases -= eta * derivative
  

class Network():
    def __init__(self):
        self.layer1 = Layer(15, 784)
        self.layer0 = Layer(10, 15)
        self.eta = 0.1

    def set_layers(self, sample_data):
        self.sample_data = sample_data
        self.layer1.set_value(sample_data)
        self.layer0.set_value(self.layer1.get_value())

        # print()
        # print()
        # print(self.layer0.values.reshape(10))
        # print(self.layer1.values.reshape(15))
        # print()
        # print()
        # exit()

    def get_prediction(self, input):
        self.set_layers(input)
        val = self.layer0.get_value()
        return np.where(val == np.amax(val))[0][0]

    def backtrack(self, y):
        w0_derivative = np.zeros(150).reshape(10, 15)
        b0_derivative = np.zeros(10).reshape(10, 1)
        a1_derivative = np.zeros(15)

        # a1 is 15 x 1 matrix
        a1 = np.transpose(self.layer1.get_value())

        # print()
        # print()
        # print(a1)
        # print()
        # print()

        for i in range(10):

            # temp is scalar
            temp = 2 * (self.layer0.get_value()[i] - y[i]) * sigmoid_derivative(self.layer0.get_z(i))

            w0_derivative[i] = a1 * temp
            b0_derivative[i] = temp
            a1_derivative += (temp * self.layer0.get_weights(i)).reshape(15)

        # print(w0_derivative)

        # print()
        # print()
        # print(a1_derivative)
        # print()
        # print()
        
        self.layer0.update_weights(self.eta, w0_derivative)
        self.layer0.update_biases(self.eta, b0_derivative)


        w1_derivative = np.zeros(11760).reshape(15, 784)
        b1_derivative = np.zeros(15).reshape(15, 1)
        
        # a2 is 784 x 1 matrix
        a2 = self.sample_data.reshape(784)

        for i in range(15):
            # temp is scalar
            temp = 2 * (-self.eta * a1_derivative[i]) * sigmoid_derivative(self.layer1.get_z(i))

            # print(a2 * temp)
            # print()
            # print(self.layer1.get_z(i))
            # print(sigmoid_derivative(self.layer1.get_z(i)))

            w1_derivative[i] = a2 * temp
            b1_derivative[i] = temp

        np.set_printoptions(threshold=sys.maxsize)

        # print()
        # print()
        # print(w1_derivative)
        # print()
        # print()

        self.layer1.update_weights(self.eta, w1_derivative)
        self.layer1.update_biases(self.eta, b1_derivative)

    def test_data(self, test_size, test_data, test_labels):
        correct_predictions = 0
        for i in range(test_size):
            if self.get_prediction(test_data[i].reshape(784, 1)) == test_labels[i]:
                correct_predictions += 1

        return correct_predictions * 100 / test_size

    def print_wb(self):
        print("\nWeights of last layer")
        for i in range(10):
            print([round(i, 2) for i in list(self.layer0.get_weights(i))])

        print("\nBiases of last layer")
        print([round(i, 2) for i in list(self.layer0.biases.reshape(10))])
        



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
    # 784 (28 * 28) integers(0 - 255)
    # each element in train_label is an integer(0 - 9)
    train_data = data[0][:train_size]
    train_lables = data[1][:train_size]
    test_data = data[0][-test_size:]
    test_lables = data[1][-test_size:]


    nn = Network()

    epochs = 10

    for epoch in range(epochs):
        for i in range(batches):
            batch_data = train_data[batch_size * i : batch_size * (i + 1)]
            batch_labels = train_lables[batch_size * i : batch_size * (i + 1)]

            for j in range(batch_size):
                sample_data = batch_data[j].reshape(784, 1)
                nn.set_layers(sample_data)

                y = np.zeros(10).reshape(10, 1)
                y[batch_labels[j] - 1] = 1.0

                nn.backtrack(y)

                # print("Sample data:")
                # print(sample_data)
                # print("sample_label:", batch_labels[j])
                # show_image(train_data[j])

    # nn.print_wb()


    accuracy = nn.test_data(test_size, test_data, test_lables)
    print("\nsample size =", sample_size)
    print("train size =", train_size)
    print("batch size =", batch_size)
    print("epochs =", epochs)
    print("accuracy of the model is:", str(accuracy) + "%")



    # view images -
    # for i in range(len(train_data)):
    #     print(train_lables[i])
    #     show_image(train_data[i])

    stop = timeit.default_timer()
    print("\nruntime:", round(stop - start, 3), 'seconds\n')