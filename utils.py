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
def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n
			
def train_val_data_split(X, Y, train_val_split):
	"""Split training data into training and validation
	"""
	N = X.shape[0]
	perm = np.random.permutation(N)
	train_inds = perm[0:int(train_val_split*N)]
	X_train = X[train_inds,:,:,:]
	Y_train = Y[train_inds]
	val_inds = perm[int(train_val_split*N):]
	X_val = X[val_inds,:,:,:]
	Y_val = Y[val_inds]
	return X_train, Y_train, X_val, Y_val

def generate_dataset_cifar10(path, n_batches=5, flatten=False):
	#shape = [3, 32, 32]
	labels = np.zeros((10000*n_batches, 10))
	features = np.zeros([10000*n_batches] + [32, 32, 3])
	for i in range(1, n_batches+1):
		file_name = path + "/data_batch_{}".format(i)
		with open(file_name, 'rb') as fo:
			#dict = pickle.load(fo, encoding="latin-1")
			dict = pickle.load(fo, encoding="latin1")

		features_batch = dict["data"]
		if not flatten: 
			features_batch = features_batch.reshape([10000] + [3, 32, 32])
			features_batch = np.rollaxis(features_batch, 1, 4)
			features[10000*(i-1):10000*i, :, : :] = features_batch
		else:
			pass
			
		labels_val = dict["labels"]

		labels[np.arange(10000*(i-1), 10000*i), labels_val] = 1
	
	return features, labels
	
def load_dataset(dataset):
	if dataset == "cifar10":
		return generate_dataset_cifar10("../datasets/cifar-10-batches-py")
	elif dataset == "tiny200":
		class_names, X_train, y_train, X_val, y_val = load_tiny_imagenet(path="../datasets/tiny-imagenet-200/", wnids_path="", resize='False', num_classes=200, dtype=np.float32)
		
		N = X_train.shape[0]
		Y_train = np.zeros((N, 200))
		Y_train[np.arange(N), y_train] = 1
		
		N = X_val.shape[0]
		Y_val = np.zeros((N, 200))
		Y_val[np.arange(N), y_val] = 1
		
		return X_train, Y_train, X_val, Y_val

def save_any(file_name, data):
	if not os.path.isdir("tests"):
		os.mkdir("tests")
	#path = os.path.join("models", model_name, "{}.txt".format(file_name))
	path = os.path.join("tests", file_name + ".txt")
	with open(path, "wb") as f:
		pickle.dump(data, f)

def load_any(file_name):
	path = os.path.join("tests", file_name + ".txt")
	with open(path, "rb") as f:
		data = pickle.load(f)
	return data

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
	n = 450
	#losses["train"] = moving_average(losses["train"], n=44950)
	#costs["train"] = moving_average(costs["train"], n=44950)
	#accs["train"] = moving_average(accs["train"], n=44950)
	
	epochs = 100
	N = 45000
	batch_size = 100
	batch_runs = int(N/batch_size)
	l_train = [0]*epochs
	c_train = [0]*epochs
	a_train = [0]*epochs
	for epoch in range(1, epochs+1):
		from_ind = (epoch-1)*batch_runs
		to_ind = epoch*batch_runs
		#print("from {} to {}".format(from_ind, to_ind))
		#print(np.mean(losses["train"][from_ind:to_ind]))
		l_train[epoch-1] = np.mean(losses["train"][from_ind:to_ind])
		c_train[epoch-1] = np.mean(costs["train"][from_ind:to_ind])
		a_train[epoch-1] = np.mean(accs["train"][from_ind:to_ind])
	
	losses["train"] = l_train
	costs["train"] = c_train
	accs["train"] = a_train
	
	plt.subplot(rows, cols, 1)
	plt.plot(losses["train"])
	plt.plot(losses["validation"])
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.subplot(rows, cols, 2)
	plt.plot(costs["train"])
	plt.plot(costs["validation"])
	plt.ylabel("cost")
	plt.xlabel("epoch")
	plt.subplot(rows, cols, 3)
	plt.plot(accs["train"])
	plt.plot(accs["validation"])
	plt.ylabel("accuracy")
	plt.xlabel("epoch")
	plt.show()
	
def plot_image(img):
	print(img)
	print(img.shape)
	img = img.reshape(32, 32, 3)
	img = img/255
	plt.imshow(img, cmap="gray")
	plt.show()
		
def plot_exp_learning_rate(losses, learning_rates):
	plt.subplot(1, 2, 1)
	plt.plot(learning_rates)
	plt.subplot(1, 2, 2)
	plt.plot(losses)
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