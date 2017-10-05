import perceptron

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
	inputs1 = [1,2]
	desired1 = [0.7]
	inputs2 = [1,2]
	desired2 = [0.7]

	logging.debug(test_net.output_vector(inputs1))
	logging.debug(test_net.output_deltas(inputs1, desired1))

	for i in range(15):
		test_net.train(inputs1, desired1)
	for i in range(15):
		test_net.train(inputs2, desired2)

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
	inputs1 = [1,2]
	desired1 = [0.7]
	inputs2 = [1,2]
	desired2 = [0.7]

	logging.debug(test_net.output_vector(inputs1))
	logging.debug(test_net.output_deltas(inputs1, desired1))

	for i in range(15):
		test_net.train(inputs1, desired1)
		test_net.train(inputs2, desired2)

if __name__ == '__main__':
	method1()
	method2()