import random, math, logging
import numpy as np

# Logging Stuff
loglevel = logging.DEBUG
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
		return self.activation_function(self.activity_function(input_vector))

	def update_weights(self, delta, input_vector):
		self.weights = [w+self.eta*delta*i for i,w in zip(input_vector, self.weights)]
		# self.bias += self.eta * delta

	def train(self, input_vector, desired):
		y = self.output(input_vector)
		delta = y*(1 - y)*(desired - y)
		# logging.debug('delta: ' + str(delta))
		self.update_weights(delta, input_vector)
		return y

class PerceptronLayer():

	def __init__(self,perceptron_list, output_flag):
		self.nodes = list(perceptron_list)
		self.output_flag = output_flag

	def error_vector(self, desired_output, output):
		if output_flag:
			return np.subtract(desired_output, output)

	def output_vector(self, input_vector):
		return [node.output(input_vector) for node in nodes]

	def output_deltas(self, desired_vector):
		return [y*(1 - y)*(desired - y) for y,desired in zip(output_vector, desired_vector)]

class PerceptronNet():

	def __init__(self,layer_list):
		self.layers = list(layer_list)
		for i in range(0, len(self.layers)):
			self.layers[i].output_flag = False
		self.layers[-1].output_flag = True

if __name__ == '__main__':
	lonely_bob = Perceptron(weights=[-0.3, 0.6], input_count=2, eta=1, bias=0.2)
	inputs1 = [1,0]
	# inputs2 = [0.2,0.3]
	# x1 = 0.8
	# x2 = 0.9
	desired1 = 0.8
	# output2 = 1
	# for i in range(0,900000):
	# 	y1 = lonely_bob.train(inputs1, output1)
	# 	y2 = lonely_bob.train(inputs2, output2)
	# 	if i % 1000 == 0:
	# 		logging.debug('y= ' + str(y1) + ' w1 = ' + str(lonely_bob.weights[0]) + ' w2 = ' + str(lonely_bob.weights[1]))
	# 		logging.debug('y= ' + str(y2) + ' w1 = ' + str(lonely_bob.weights[0]) + ' w2 = ' + str(lonely_bob.weights[1]))

	for i in range(0,101):
		y = lonely_bob.train(inputs1, desired1)
		if i % 100 == 0:
			logging.debug('y= ' + str(y) + ' w1 = ' + str(lonely_bob.weights[0]) + ' w2 = ' + str(lonely_bob.weights[1]))
	logging.debug('output1: ' + str(lonely_bob.output(inputs1)))
	# logging.debug('output2: ' + str(lonely_bob.output(inputs2)))