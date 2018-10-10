import struct
import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
	result = 1
	for x in range(1, n+1):
		result *= x
	return result

def nCr(n, r):
	return factorial(n) / (factorial(r) * factorial(n-r))


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
		imgs = np.zeros( (num_imgs, n_row, n_col) )
		for i in range(num_imgs):
			for row in range(n_row):
				for col in range(n_col):
					value = struct.unpack('>B', f.read(1))[0]
					imgs[i][row][col] = value
		return imgs

def read_mnist_label(file_name):
	with open(file_name, 'rb') as f:
		magic_number, num_labels = struct.unpack('>ii', f.read(8))
		labels = np.zeros(num_labels)
		for i in range(num_labels):
			value = struct.unpack('>B', f.read(1))[0]
			labels[i] = value
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



class Feature_Bins():
	def __init__(self, n_bins=32, id='',  min_count=10, min_value=0, max_value=256):
		self.bins = [min_count for x in range(n_bins)]
		self.id = id
		self.min = min_value
		self.max = max_value

	def pseudocount(self, min_count=10):
		'''  avoid empty bin '''
		for i, count in enumerate(self.bins):
			if count < min_count:
				diff = min_count - count
				self.bins[i] = min_count
				self.bins[argmax(self.bins)] -= diff
		return self

	def get_bin_num(self, value):
		interval = (self.max-self.min) // len(self.bins)
		c = value // interval
		return c


	def to_bin(self, value):
		interval = (self.max-self.min) // len(self.bins)
		c = value // interval
		self.bins[int(c)] += 1

	def get_count(self, bin_num):
		return self.bins[int(bin_num)]
	def total_count(self):
		return sum(self.bins)

	def __len__(self):
		return len(self.bins)

	''' define what will A[i] return '''
	def __getitem__(self,key):
		return self.bins[key]

	''' define what will happen when A[i] = k is called '''
	def __setitem__(self,key,value):
		self.bins[key] = value 

	def __str__(self):
		s = 'Feature Bins %s \n' %(self.id)
		num_space = [max(len(str(i)), len(str(count)))+1 for i, count in enumerate(self.bins)] 
		first_line = [ '{bin_id:>{width}}'.format(bin_id=i, width=num_space[i]) for i in range(len(self.bins)) ]
		second_line = [ '{bin_count:>{width}}'.format(bin_count=count, width=num_space[i]) for i, count in enumerate(self.bins) ]
		s += ''.join(first_line) + '\n'
		s += ''.join(second_line) + '\n'
		return s


def print_probs(probs):
	first_line = [ '{label:>5}'.format(label=i) for i in range(len(probs)) ]
	second_line = [ '{prob:>5}'.format(prob='%.2f'%prob) for prob in probs ]
	s = '{head:<6}'.format(head='Label') + ''.join(first_line) + '\n' 
	s += '{head:<6}'.format(head='Prob') + ''.join(second_line) + '\n' 
	print(s)

if __name__ == '__main__':
	bins = Feature_Bins(32)
	print(len(str(10)))
	print(bins)
	probs = [0.1, 0.5888, 0.11111111, 0.124556]
	print_probs(probs)
	l = [9.640073411041338e-307, 0.0, 3.782100448838266e-290, 2.2575530836797617e-276, 3.858000729702203e-259, 3.0518926958908267e-269, 1.583220655e-315, 1.1855717910607961e-205, 1.206435365501705e-274, 1.054756447106051e-237]
	print('%.12f' %sum(l))