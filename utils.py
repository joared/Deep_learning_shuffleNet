import os

import tensorflow as tf

import numpy as np

import pickle
import matplotlib.pyplot as plt

from data_utils import load_tiny_imagenet
"""
def generate_dataset_tiny_200(dir="../dataset/tiny-imagenet-200", flatten=False)
	dir = os.path.join(dir, "tiny-imagenet-200")
	wnids_path = os.path.join(dir, "wnids.txt")
	words_path = os.path.join(dir, "words.txt")
	
	train_path = os.path.join(dir, "train")
	val_path = os.path.join(dir, "val")
	test_path = os.path.join(dir, "test")
	
	
	with open(wnids, "r") as f:
		im_path = ""
"""

def train_val_data_split(X, Y, train_val_split):
	"""Split training data into training and validation
	"""
	N = X.shape[0]
	perm = np.random.permutation(N)
	train_inds = perm[0:int(train_val_split*N)]
	X = X[train_inds,:,:,:]
	Y = Y[train_inds]
	val_inds = perm[int(train_val_split*N):]
	X_val = X[val_inds,:,:,:]
	Y_val = Y[val_inds]
	return X, Y, X_val, Y_val

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
	
	return features, labels
	
def load_dataset(dataset):
	if dataset == "cifar10":
		return generate_dataset_cifar10("../datasets/cifar-10-batches-py/data_batch_1")
	elif dataset == "tiny200":
		class_names, X_train, y_train, X_val, y_val = load_tiny_imagenet(path="../datasets/tiny-imagenet-200/", wnids_path="", resize='False', num_classes=200, dtype=np.float32)
		
		N = X_train.shape[0]
		Y_train = np.zeros((N, 200))
		Y_train[np.arange(N), y_train] = 1
		
		N = X_val.shape[0]
		Y_val = np.zeros((N, 200))
		Y_val[np.arange(N), y_val] = 1
		
		return X_train, Y_train, X_val, Y_val

def load_training_data(model_name):
	path = os.path.join("models", model_name, "training_data.txt")
	with open(path, "rb") as f:
		data = pickle.load(f)
	return data
	
def save_training_data(data, model_name):
	if not os.path.isdir("models"):
		os.mkdir("models")
	if not os.path.isdir(os.path.join("models", model_name)):
		os.mkdir(os.path.join("models", model_name))
	path = os.path.join("models", model_name + "/training_data.txt")
	with open(path, "wb") as f:
		pickle.dump(data, f)
	
def plot_training_data(losses, costs, accs, model_name=None):
	rows = 1
	cols = 3
	plt.subplot(rows, cols, 1)
	plt.plot(losses["train"])
	plt.plot(losses["validation"])
	plt.ylabel("loss")
	plt.xlabel("epoch")
	#plt.subplot(rows, cols, 2)
	#plt.plot(accs["train"])
	#plt.plot(accs["validation"])
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.show()
		
def analyze_model(model):
	pass
	"""
	import tensorflow.python.framework.ops as ops
	model() # builds the graph
	n_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
	print("Trainable variables: ", n_vars)
	#for v in tf.trainable_variables():
	#	print(v)
	g = tf.get_default_graph()
	run_meta = tf.RunMetadata()
	opts = tf.profiler.ProfileOptionBuilder.float_operation()
	flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
	if flops is not None:
		print('TF stats gives',flops.total_float_ops)
	"""
	
if __name__ == "__main__":
	pass
	#import numpy
	#print("OK")