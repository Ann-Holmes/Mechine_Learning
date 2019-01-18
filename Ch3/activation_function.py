import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策, 所有的整数都减去最大的数, 再求指数
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)
    y4 = softmax(x)
    plt.plot(x, y1, )
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.plot(x, y4)
    plt.show()


if __name__ == "__main__":
    main()
