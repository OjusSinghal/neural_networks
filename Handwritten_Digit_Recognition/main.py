import gzip
import numpy as np
import matplotlib.pyplot as plt


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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer():
    def __init__(self, n, prev):
        self.n = n
        self.values = np.zeros(n).reshape(n, 1)
        self.biases = np.zeros(n).reshape(n, 1)
        self.weights = np.zeros(n * prev).reshape(n, prev)

    def set_value(self, pre_layer):
        self.values = np.dot(self.weights, pre_layer) + self.biases
        self.values = sigmoid(self.values)

    def get_value(self):
        return self.values

    

class Network():
    def __init__(self):
        self.layer1 = Layer(15, 784)
        self.layer0 = Layer(10, 15)

    def set_layers(self, input):
        self.layer1.set_value(input)
        self.layer0.set_value(self.layer1.get_value())

    def get_prediction(self, input):
        val = layer0.get_value()
        return np.where(val == np.amax(val))[0][0]

    def get_output(self):
        return layer0.get_value()


if __name__ == "__main__":
    sample_size = 60000
    train_size = 50000
    test_size = sample_size - train_size

    data = get_data(sample_size)
    
    # each element in train_data is a 1D array of 784 (28 * 28) integers(0 - 255)
    # each element in train_label is an integer(0 - 9)
    train_data = data[0][:train_size]
    train_lables = data[1][:train_size]
    test_data = data[0][-test_size:]
    test_lables = data[1][-test_size:]

    batch_size = 20
    batches = int(train_size / batch_size)

    nn = Network()

    for i in range(batches):
        batch_data = train_data[20 * i : 20 * (i + 1)]
        batch_label = train_lables[20 * i : 20 * (i + 1)]

        cost_gradient_weights = np.zeros(2).reshape(2, 1)
        cost_gradient_biases = np.zeros(2).reshape(2, 1)

        for j in range(batch_size):
            sample_data = batch_data[j].reshape(784, 1)
            nn.set_layers(sample_data)
            y = np.zeros(10).reshape(10, 1)
            y[batch_label[j]] = 1.0


            # update cost gradients

        
            





    # view images -
    # for i in range(len(train_data)):
    #     print(train_lables[i])
    #     show_image(train_data[i])


