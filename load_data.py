import tensorflow.examples.tutorials.mnist.input_data as input_data
def load_data(train_set=5000,test_set=2000):
	"""
	Input:
		train_set: the quantities of train_set
		test_set:  the quantities of test_set

	Output:
		train_x (28*28,train_set)
		train_y (1,train_set)
		test_x  (28*28,test_set)
		test_y  (1,test_set)
	"""
	mnist = input_data.read_data_sets("datasets/", one_hot=True)

	train_x_orig,train_y_orig,test_x_orig,test_y_orig=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

	train_x=train_x_orig[0:train_set].T
	train_y=train_y_orig[0:train_set].T
	test_x=test_x_orig[0:test_set].T
	test_y=test_y_orig[0:test_set].T
	print("The shape of train_x is ("+ str(train_x.shape[0])+', '+str(train_x.shape[1])+')')
	print("The shape of train_y is ("+ str(train_y.shape[0])+', '+str(train_y.shape[1])+')')
	print("The shape of test_x is ("+ str(test_x.shape[0])+', '+str(test_x.shape[1])+')')
	print("The shape of test_y is ("+ str(test_y.shape[0])+', '+str(test_y.shape[1])+')')
	return train_x,train_y,test_x,test_y

