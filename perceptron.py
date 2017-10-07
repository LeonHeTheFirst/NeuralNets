import random, math, logging
import numpy as np

# Logging Stuff
loglevel = logging.INFO
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=loglevel)
logger = logging.getLogger(__name__)

class Perceptron():

	# TODO: Set seed for random generation
	def __init__(self, input_count, weights=None, eta=1, bias=0):
		if weights == None:
			self.weights = [random.uniform(-1, 1) for i in range(0,input_count)]
		else:
			self.weights = list(weights)
		self.bias = bias
		self.eta = eta

	def activity_function(self, inputs):
		s = 0
		for i,w in zip(inputs, self.weights):
			s += i*w
		return s

	def activation_function(self, A):
		v = 1/(1+math.exp(-1 * A))
		return 1/(1+math.exp(-1 * (A + self.bias)))

	def output(self, input_vector):
		activity = self.activity_function(input_vector)
		logging.debug('activity function output: ' + str(activity))
		activation = self.activation_function(activity)
		logging.debug('activation function output: ' + str(activation))
		return self.activation_function(activity)

	def calculate_delta(self, output, delta_vector, weights):
		sigma = 0
		for delta, weight in zip(delta_vector, weights):
			sigma += delta * weight
		return (1 - output) * output * sigma

	def update_weights(self, delta, input_vector):
		self.weights = [w+self.eta*delta*i for i,w in zip(input_vector, self.weights)]
		self.bias += self.eta * delta

	# def train(self, input_vector, desired):
	# 	y = self.output(input_vector)
	# 	delta = y*(1 - y)*(desired - y)
	# 	self.update_weights(delta, input_vector)
	# 	return y

class PerceptronLayer():

	def __init__(self, input_count=None, node_count=None, eta=1, perceptron_list=None, output_flag=False):
		if perceptron_list is not None:
			self.nodes = list(perceptron_list)
		elif input_count is not None and node_count is not None:
			self.nodes = []
			for x in range(node_count):
				self.nodes.append(Perceptron(input_count=input_count, eta=eta))
		else:
			logger.error('Incorrect arguments to PerceptronLayer')
		self.output_flag = output_flag

	def node_count(self):
		return len(self.nodes)

	def error_vector(self, desired_output, output):
		if self.output_flag:
			return np.subtract(desired_output, output)

	def input_vector_size():
		return len(nodes[0].weights)

	def output_vector(self, input_vector):
		return [node.output(input_vector) for node in self.nodes]

	def output_deltas(self, output_vector, desired_vector):
		return [y*(1 - y)*(desired - y) for y,desired in zip(output_vector, desired_vector)]

	def calculate_deltas(self, output_vector, next_delta_vector, weights_vectors):
		delta_vector = []
		logging.debug('Calculating deltas, nodes size: ' + str(len(self.nodes)))
		for i in range(0, len(output_vector)):
			delta_vector.append(self.nodes[i].calculate_delta(output_vector[i],
															 next_delta_vector,
															 weights_vectors[i]))
		return delta_vector

	# The vector returned by this function will be of form [[w_11, w_21], [w_12, w_22], [w_13, w_23]]
	# and not of the form [[w_11, w_12, w_13], [w_21, w_22, w_23]]
	def get_weights_vector(self):
		weights_vector = []
		# First get the list of lists in the opposite format
		for node in self.nodes:
			weights_vector.append(node.weights)
		return weights_vector

	# The vector returned by this function will be of form [[w_11, w_12, w_13], [w_21, w_22, w_23]]
	# and not of the form [[w_11, w_21], [w_12, w_22], [w_13, w_23]]
	def get_weights_vector_transposed(self):
		# Transpose get_weights_vector() to the format we want
		weights_vector = list(map(list, zip(*self.get_weights_vector())))
		return weights_vector

	def update_weights(self, delta_vector, input_vector):
		logging.debug('Updating weights, node vector size: ' + str(self.node_count()))
		for i in range(0, self.node_count()):
			self.nodes[i].update_weights(delta_vector[i], input_vector)

class PerceptronNet():

	def __init__(self,layer_list):
		self.layers = list(layer_list)
		for i in range(0, len(self.layers)):
			self.layers[i].output_flag = False
		self.layers[-1].output_flag = True

	def save_network(self):
		# TBD
		pass

	def layer_count(self):
		return len(self.layers)

	def input_vector_size():
		return layer_list[0].input_vector_size()

	def output_vector(self, input_vector):
		next_input_vector = input_vector
		for layer in self.layers:
			next_input_vector = layer.output_vector(next_input_vector)
		return next_input_vector

	def error_vector(self, input_vector, desired_output_vector):
		next_input_vector = input_vector
		for layer in self.layers:
			next_input_vector = layer.output_vector(next_input_vector)
		return [(desired - output) for desired, output in zip(desired_output_vector, next_input_vector)]

	def output_deltas(self, input_vector, desired_vector):
		output_vector = self.output_vector(input_vector)
		return [y*(1 - y)*(desired - y) for y,desired in zip(output_vector, desired_vector)]

	def train(self, input_vector, desired_output_vector):
		# First find the input vectors at each level
		input_vectors = [[] for i in range(0, self.layer_count())]
		input_vectors[0] = input_vector
		for i in range(1, self.layer_count()):
			input_vectors[i] = self.layers[i-1].output_vector(input_vectors[i-1])
		# Next find the delta vectors at each layer going backwards
		layer_deltas = [[] for i in range(0, self.layer_count())]
		layer_deltas[self.layer_count()-1] = self.output_deltas(input_vector, desired_output_vector)
		for i in reversed(range(0, self.layer_count()-1)):
			layer_deltas[i] = self.layers[i].calculate_deltas(input_vectors[i+1],
															  layer_deltas[i+1],
															  self.layers[i+1].get_weights_vector_transposed())
			logging.debug(layer_deltas[i])
		# Lastly, update the weights at each layer
		for i in range(0, self.layer_count()):
			self.layers[i].update_weights(layer_deltas[i], input_vectors[i])

def test_network():
	first_layer = []
	second_layer = []
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	second_layer.append(Perceptron(weights=[0.8, 0.8], input_count=2, eta=1, bias=0))
	layer_list = []
	layer_list.append(PerceptronLayer(perceptron_list=first_layer, output_flag=False))
	layer_list.append(PerceptronLayer(perceptron_list=second_layer, output_flag=True))
	test_net = PerceptronNet(layer_list)
	inputs1 = [1,2]
	desired1 = [0.7]
	logging.info(test_net.output_vector(inputs1)) # Should be [0.757223870792493]
	logging.info(test_net.output_deltas(inputs1, desired1)) # Should be [-0.01051980066099822]
	for x in range(10):
		test_net.train(inputs1, desired1)
	logging.info(test_net.output_vector(inputs1)) # Should be [0.723439394911163]


if __name__ == '__main__':
	first_layer = []
	second_layer = []
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	second_layer.append(Perceptron(weights=[0.8, 0.8], input_count=2, eta=1, bias=0))
	layer_list = []
	layer_list.append(PerceptronLayer(perceptron_list=first_layer, output_flag=False))
	layer_list.append(PerceptronLayer(perceptron_list=second_layer, output_flag=True))
	test_net = PerceptronNet(layer_list)
	inputs1 = [1,2]
	desired1 = [0.7]
	inputs2 = [-1,-1]
	desired2 = [0.05]

	# Method 1
	for x in range(15):
		test_net.train(inputs1, desired1)
		test_net.train(inputs2, desired2)
		logging.info(test_net.output_vector(inputs1))
		logging.info(test_net.output_vector(inputs2))
	# Method 2
	for x in range(15):
		test_net.train(inputs1, desired1)
	for x in range(15):
		test_net.train(inputs2, desired2)
		logging.info(test_net.output_vector(inputs1))
		logging.info(test_net.output_vector(inputs2))
	# logging.info(test_net.output_vector(inputs1))
	# logging.info(test_net.output_deltas(inputs1, desired1))
	# for x in range(10):
	# 	test_net.train(inputs1, desired1)
	# logging.info(test_net.output_vector(inputs1))
