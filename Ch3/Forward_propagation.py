import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import perceptron as pp


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1) + b1
    z1 = pp.sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = pp.sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y
