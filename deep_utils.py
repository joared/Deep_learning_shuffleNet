import tensorflow as tf

import numpy as np

import pickle
import matplotlib.pyplot as plt

def generate_dataset_cifar10(file_name, flatten=False):
	with open(file_name, 'rb') as fo:
		#dict = pickle.load(fo, encoding="latin1")
		dict = pickle.load(fo, encoding="latin1")

	features = dict["data"]
	if not flatten:
		features = features.reshape(10000, 32, 32, 3)
	labels_val = dict["labels"]

	labels = np.zeros((10000, 10))
	labels[np.arange(10000), labels_val] = 1

	print(labels_val[0:100])
	print(labels[0:10])
	
	return features, labels

def plot(x_data):
	x_data = x_data.reshape([10000, 3, 32, 32]).transpose(0,2,3,1).astype("uint8")
	im = x_data[0]
	plt.imshow(im)
	plt.show()
		
def save_model(session, export_dir, x, y, y_pred, loss, optimizer, learning_rate):
	inputs = {"x": x, "y": y, "learning_rate": learning_rate}
	outputs = {"y_pred": y_pred, "loss": loss, "optimizer": optimizer}
	x, y, y_pred, loss, optimizer, learning_rate
	tf.saved_model.simple_save(session, export_dir, inputs, outputs)
	
def load_model(session, export_dir):
	pass
		
def test():
	import sys
	sys.path.append("C:\\Users\\Joris\\deep_learning\\tensorpack")
	from shufflenet import Model
	import tensorflow as tf
	image = tf.placeholder(tf.float64, shape=[224,224,3])
	args = None
	m = Model().get_logits(image)
	print(m)
	
if __name__ == "__main__":
	test()
	#import numpy
	#print("OK")
	
	

