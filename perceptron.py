import random, math, logging, numpy

# Logging Stuff
loglevel = logging.DEBUG
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT, level=loglevel)
logger = logging.getLogger(__name__)

class Perceptron():

	def __init__(self, input_count, weights=None, eta=1, bias=0):
		if weights == None:
			self.weights = [random.uniform(-1, 1) for i in range(0,input_count)]
		else:
			self.weights = [w for w in weights]
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

	def output(self, inputs):
		return self.activation_function(self.activity_function(inputs))

	def update_weights(self, delta, inputs):
		self.weights = [w+self.eta*delta*i for i,w in zip(inputs, self.weights)]
		self.bias += 0 # self.eta * delta

	def train(self, inputs, desired):
		y = self.output(inputs)
		delta = y*(1 - y)*(desired - y)
		# logging.debug('delta: ' + str(delta))
		self.update_weights(delta, inputs)
		return y

class PerceptronLayer():

	def __init__(self,node_count):
		self.nodes = [Perceptron()]

class PerceptronNet():

	def __init__(self,layer_count):
		pass

if __name__ == '__main__':
	lonely_bob = Perceptron(weights=[-0.3, 0.6], input_count=2, eta=0.1, bias=0.2)
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