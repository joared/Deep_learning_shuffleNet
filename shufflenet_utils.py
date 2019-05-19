import os

import tensorflow as tf

import numpy as np

import pickle
import matplotlib.pyplot as plt

from shufflenet_models import Model
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
		return load_tiny_imagenet(path="../datasets/tiny-imagenet-200/tiny-imagenet-200", wnids_path="", resize='False', num_classes=200, dtype=np.float32)
	
def save_training_data(data, model_name):
	if not os.path.isdir("models"):
		os.mkdir("models")
	if not os.path.isdir(os.path.join("models", model_name)):
		os.mkdir(os.path.join("models", model_name))
	path = os.path.join("models", model_name + "/training_data.txt")
	with open(path, "wb") as f:
		pickle.dump(data, f)

def load_training_data(model_name):
	path = os.path.join("models", model_name, "training_data.txt")
	with open(path, "rb") as f:
		data = pickle.load(f)
	return data
	
def plot_training_data(*model_names):
	rows = 1 # len(model_names)
	cols = 2
	for i, model_name in enumerate(model_names):
		losses, accs = load_training_data(model_name)
		
		plt.subplot(rows, cols, 1)
		plt.plot(losses["train"], label=model_name)
		plt.ylabel("loss")
		plt.xlabel("epoch")
		plt.subplot(rows, cols, 2)
		plt.plot(accs["train"], label=model_name)
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