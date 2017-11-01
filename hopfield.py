import random, math, logging
import numpy as np

# Logging Stuff
loglevel = logging.INFO
FORMAT = '%(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=loglevel)
logger = logging.getLogger(__name__)

class HopfieldNet():
	def __init__(self, size):
		self.matrix = 0

	def layer_count(self):
		pass

	def input_vector_size():
		return self.matrix.x

	def output_vector(self, input_vector):
		output = self.matrix * input_vector
		return output

	def error_vector(self, input_vector, desired_output_vector):
		pass

	def output_deltas(self):
		pass

	def train(self, exemplar):
		pass

	def big_e(self, input_vector, desired_output_vector):
		pass

	def get_matrix(self):
		return self.matrix

def test_network():
	pass


if __name__ == '__main__':
	pass