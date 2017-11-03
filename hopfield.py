import random, math, logging
import numpy as np

# Logging Stuff
loglevel = logging.INFO
FORMAT = '%(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=loglevel)
logger = logging.getLogger(__name__)

class HopfieldNet():
	def __init__(self, size):
		self.matrix = np.asmatrix(np.zeros(shape = (size, size)))

	def layer_count(self):
		pass

	def get_size(self):
		return self.matrix.shape[0]

	def output_vector(self, input_vector):
		history = []
		next_output = np.asmatrix(input_vector).transpose()
		print('input: ', next_output)
		while not any((next_output == mat).all() for mat in history[:-1]):
			next_output = np.matmul(self.matrix, next_output)
			for i, value in np.ndenumerate(next_output):
				if value > 1:
					next_output[i] = 1
				elif value < -1:
					next_output[i] = -1
			history.append(next_output.copy())
			print(next_output)
		print(next_output)
		return next_output.transpose()

	def error_vector(self, input_vector, desired_output_vector):
		pass

	def output_deltas(self):
		pass

	def train(self, exemplar):
		"""exemplar must be np.array()"""
		new = np.matmul(np.asmatrix(exemplar).transpose(), np.asmatrix(exemplar)) - np.identity(self.get_size())
		self.matrix += new

	def big_e(self, input_vector, desired_output_vector):
		pass

	def get_matrix(self):
		return self.matrix

def test_network():
	testnet = HopfieldNet(9)
	for i in range(10):
		testnet.train([1,1,1,1,1,1,1,1,1])
	print(testnet.matrix)
	testnet.output_vector([-1,1,1,-1,1,-1,1,1,-1])


if __name__ == '__main__':
	test_network()