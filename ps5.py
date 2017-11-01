from perceptron import Perceptron, PerceptronLayer, PerceptronNet
import math, logging

# Logging Stuff
logger = logging.getLogger(__name__)

num_iterations = 100

def method1():
	first_layer = []
	second_layer = []
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	second_layer.append(Perceptron(weights=[0.8, 0.8], input_count=2, eta=1, bias=0))
	layer_list = []
	layer_list.append(PerceptronLayer(perceptron_list=first_layer, output_flag=False))
	layer_list.append(PerceptronLayer(perceptron_list=second_layer, output_flag=True))
	test_net = PerceptronNet(layer_list)
	inputs1 = [1,1]
	desired1 = [0.9]
	inputs2 = [-1,-1]
	desired2 = [0.05]

	# logger.info(test_net.output_deltas(inputs1, desired1))

	for i in range(num_iterations):
		test_net.feed_forward_back_propagation(inputs1, desired1)
		test_net.feed_forward_back_propagation(inputs2, desired2)

	logger.info('Output from input 1: ' + str(test_net.output_vector(inputs1)))
	logger.info('Output from input 2: ' + str(test_net.output_vector(inputs2)))
	logger.info('Network Weights for Method 1: ' + str(test_net.get_all_weights()))
	logger.info('Error for Method 1, first pair: ' + str(test_net.big_e(inputs1, desired1)))
	logger.info('Error for Method 1, second pair: ' + str(test_net.big_e(inputs2, desired2)))


def method2():
	first_layer = []
	second_layer = []
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	first_layer.append(Perceptron(weights=[0.3, 0.3], input_count=2, eta=1, bias=0))
	second_layer.append(Perceptron(weights=[0.8, 0.8], input_count=2, eta=1, bias=0))
	layer_list = []
	layer_list.append(PerceptronLayer(perceptron_list=first_layer, output_flag=False))
	layer_list.append(PerceptronLayer(perceptron_list=second_layer, output_flag=True))
	test_net = PerceptronNet(layer_list)
	inputs1 = [1,1]
	desired1 = [0.9]
	inputs2 = [-1,-1]
	desired2 = [0.05]

	# logger.info(test_net.output_deltas(inputs1, desired1))

	for i in range(num_iterations):
		test_net.feed_forward_back_propagation(inputs1, desired1)
	for i in range(num_iterations):
		test_net.feed_forward_back_propagation(inputs2, desired2)

	logger.info('Output from input 1: ' + str(test_net.output_vector(inputs1)))
	logger.info('Output from input 2: ' + str(test_net.output_vector(inputs2)))
	logger.info('Network Weights for Method 2: ' + str(test_net.get_all_weights()))
	logger.info('Error for Method 2, first pair: ' + str(test_net.big_e(inputs1, desired1)))
	logger.info('Error for Method 2, second pair: ' + str(test_net.big_e(inputs2, desired2)))

if __name__ == '__main__':
	method1()
	method2()