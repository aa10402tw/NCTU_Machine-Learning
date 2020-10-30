import struct
import numpy as np
import matplotlib.pyplot as plt

import math


def gamma(n):
    return factorial(n - 1)


def beta_pdf(p, a, b):
    return (p**(a - 1)) * ((1 - p)**(b - 1)) * (gamma(a + b) / (gamma(a) * gamma(b)))


def factorial(n):
    result = 1
    for x in range(1, n + 1):
        result *= x
    return result


def nCr(n, r):
    return factorial(n) / (factorial(r) * factorial(n - r))


def load_mnist(train=True):
    if train:
        imgs = read_mnist_img(file_name='data/train-images.idx3-ubyte')
        labels = read_mnist_label(file_name='data/train-labels.idx1-ubyte')
    else:
        imgs = read_mnist_img(file_name='data/t10k-images.idx3-ubyte')
        labels = read_mnist_label(file_name='data/t10k-labels.idx1-ubyte')
    return imgs, labels


def read_mnist_img(file_name):
    with open(file_name, 'rb') as f:
        magic_number, num_imgs = struct.unpack('>ii', f.read(8))
        n_row, n_col = struct.unpack('>ii', f.read(8))
        imgs = np.zeros((num_imgs, n_row, n_col))
        for i in range(num_imgs):
            for row in range(n_row):
                for col in range(n_col):
                    value = struct.unpack('>B', f.read(1))[0]
                    imgs[i][row][col] = value
        return imgs


def read_mnist_label(file_name):
    with open(file_name, 'rb') as f:
        magic_number, num_labels = struct.unpack('>ii', f.read(8))
        labels = np.zeros(num_labels).astype(np.int32)
        for i in range(num_labels):
            value = struct.unpack('>B', f.read(1))[0]
            labels[i] = int(value)
        return labels


def test_load_minst():
    imgs, labels = load_mnist(train=True)
    print(imgs.shape, labels.shape)
    img = imgs[0]
    title = labels[0]
    print(title)
    plt.imshow(img, cmap='gray'), plt.title(title)
    plt.show()


def argmax(list_):
    return max(enumerate(list_), key=lambda x: x[1])[0]


# class Feature_Bins():

#     def __init__(self, n_bins=32, id='',  min_count=10, min_value=0, max_value=256):
#         self.bins = [min_count for x in range(n_bins)]
#         self.id = id
#         self.min = min_value
#         self.max = max_value

#     def pseudocount(self, min_count=10):
#         '''  avoid empty bin '''
#         for i, count in enumerate(self.bins):
#             if count < min_count:
#                 diff = min_count - count
#                 self.bins[i] = min_count
#                 self.bins[argmax(self.bins)] -= diff
#         return self

#     def get_bin_num(self, value):
#         interval = (self.max - self.min) // len(self.bins)
#         c = value // interval
#         return c

#     def to_bin(self, value):
#         interval = (self.max - self.min) // len(self.bins)
#         c = value // interval
#         self.bins[int(c)] += 1

#     def get_count(self, bin_num):
#         return self.bins[int(bin_num)]

#     def total_count(self):
#         return sum(self.bins)

#     def __len__(self):
#         return len(self.bins)

#     ''' define what will A[i] return '''

#     def __getitem__(self, key):
#         return self.bins[key]

#     ''' define what will happen when A[i] = k is called '''

#     def __setitem__(self, key, value):
#         self.bins[key] = value

#     def __str__(self):
#         s = 'Feature Bins %s \n' % (self.id)
#         num_space = [max(len(str(i)), len(str(count))) + 1 for i, count in enumerate(self.bins)]
#         first_line = ['{bin_id:>{width}}'.format(
#             bin_id=i, width=num_space[i]) for i in range(len(self.bins))]
#         second_line = ['{bin_count:>{width}}'.format(bin_count=count, width=num_space[
#                                                      i]) for i, count in enumerate(self.bins)]
#         s += ''.join(first_line) + '\n'
#         s += ''.join(second_line) + '\n'
#         return s


def print_probs(probs):
    first_line = ['{label:>5}'.format(label=i) for i in range(len(probs))]
    second_line = ['{prob:>5}'.format(prob='%.2f' % prob) for prob in probs]
    s = '{head:<6}'.format(head='Label') + ''.join(first_line) + '\n'
    s += '{head:<6}'.format(head='Prob') + ''.join(second_line) + '\n'
    print(s)


class Gaussian():

    def __init__(self, id='', smooth=0.01):
        self.num_data = 0
        self.sum_of_square = 0
        self.sum_of_data = 0
        self.id = id
        self.smooth = smooth

    def update(self, data):
        self.num_data += 1
        self.sum_of_data += data
        self.sum_of_square += data**2

    @property
    def mean(self):
        if self.num_data == 0:
            return 0.0
        return self.sum_of_data / self.num_data

    @property
    def variance(self):
        ''' Var = 平方平均 - 平方平均 '''
        if self.num_data == 0:
            return 0.0
        return (self.sum_of_square / self.num_data) - (self.mean**2) + self.smooth

    @property
    def std(self):
        return self.variance ** (1 / 2)

    def pdf(self, x):
        p = 1 / (self.std * math.sqrt(2 * math.pi)) * \
            math.exp((-1 / 2) * ((x - self.mean) / self.std)**2)
        if p < 0.00001:
            print('Data = %d, Mean = %2.f, Var = %.2f, Prob = %f' %
                  (x, self.mean, self.variance, p))
        return max(0.0000001, p)

    def logpdf(self, x):
        return (-1 / 2) * math.log(2 * math.pi * self.variance) + -1 * ((x - self.mean)**2 / (2 * (self.variance**2)))

    def distance(self, data):
        return ((data - self.mean) / self.std)

    def __str__(self):
        s = 'Gaussian %s\n' % (self.id)
        s += 'Mean = %.2f, Variance= %.2f (#data=%d)' % (self.mean, self.variance, self.num_data)
        return s


def img_to_vector(imgs):
    N, dims = imgs.shape[0], imgs.shape[1:]
    dim = 1
    for d in dims:
        dim *= d
    X = np.reshape(imgs, (N, dim))
    return X

if __name__ == '__main__':
    # f1 = Feature_Gaussian()
    # f1.update(-1)
    # f1.update(1)
    # print(f1.mean)
    # print(f1.variance)
    # print(f1.log_pdf(50))
    # import scipy.stats
    # print(scipy.stats.norm(0, 1).logpdf(50))
    X = img_to_vector(np.zeros((10, 28, 28)))
    print(X.shape)
