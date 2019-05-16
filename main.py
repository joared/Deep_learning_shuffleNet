import tensorflow as tf

import numpy as np

import pickle
import matplotlib.pyplot as plt

from deep_utils import generate_dataset_cifar10
from shufflenet_model import model_shuffleNet_cifar10

def model_cifar10_xxxxxx():
	x = tf.placeholder(tf.float64, shape=(None, 3072))
	y = tf.placeholder(tf.float64, shape=(None, 10))
	print(x.shape)
	with tf.variable_scope("dafakh"):
		#w = tf.Variable(np.random.normal(size = [3072, 10]))
		#y_pred = tf.matmul(x, w)
		h = tf.layers.dense(x, 50)#, activation = "relu")
		h2 = tf.layers.dense(h, 50)#, activation = "relu")
		y_pred = tf.layers.dense(h2, 10)
		loss = tf.losses.softmax_cross_entropy(y, y_pred)
		#loss = tf.reduce_mean(tf.square(y_pred-y), 1)
		
	return x, y, y_pred, loss
	
def model_shuffleNet_cifar10_conv():
	x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
	y = tf.placeholder(tf.float32, shape=(None, 10))
	print(x.shape)
	with tf.variable_scope("dafakh"):
		x_norm = tf.image.per_image_standardization(x)
		conv2d_1 = tf.layers.conv2d(x_norm, 24, (3,3), strides=2, padding="same")
		conv2d_1 = tf.layers.batch_normalization(conv2d_1)
		conv2d_1 = tf.nn.relu(conv2d_1)
		
		conv2d_2 = tf.layers.conv2d(conv2d_1, 48, (2,2), strides=1, padding="same")
		conv2d_2 = tf.layers.batch_normalization(conv2d_2)
		conv2d_2 = tf.nn.relu(conv2d_2)
		
		h = tf.layers.flatten(conv2d_2)
		h = tf.layers.dense(h, 50, activation="relu")#, activation = "relu")
		
		y_pred = tf.layers.dense(h, 10)
		loss = tf.losses.softmax_cross_entropy(y, y_pred)
		#loss = tf.reduce_mean(tf.square(y_pred-y), 1)
		
	return x, y, y_pred, loss


class Model:
	def __init__(self, session=None, x=None, y=None, y_pred=None, loss=None, optimizer=None, learning_rate=None):
		self.session = session
		self.x = x
		self.y = y
		self.y_pred = y_pred
		self.loss = loss
		self.optimizer = optimizer 
		self.learning_rate = learning_rate


def compute_accuracy(y_pred_batch, y_batch):
	s = 0
	for p, y in zip(y_pred_batch, y_batch):
		p_index = np.argmax(p)
		y_index = np.argmax(y)
		if p_index == y_index:
			s += 1
			
	acc = s/y_pred_batch.shape[0]
	print("accuracy:", acc)
	return acc

def run(model, data_path="..\\datasets\\cifar-10-batches-py\\data_batch_1"):

	x_batch, y_batch = generate_dataset_cifar10(data_path)
	x, y, y_pred, loss = model()
	
	
	learning_rate = tf.placeholder(tf.float32, shape=[])
	#lr = 0.00006
	lr = 0.065
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		session.run(init)
		
		y_pred_batch = session.run(y_pred, {x: x_batch})
		compute_accuracy(y_pred_batch, y_batch)
		
		for epoch in range(200):
			perm = np.random.permutation(10000)
			batch_size = 100
			batch_nr = 1
			print("epoch", epoch)
			for i in range(int(10000/batch_size)):
				inds = perm[i*batch_size:batch_size*(i+1)]
				feed_dict = {x: x_batch[inds], y: y_batch[inds], learning_rate: lr}
				loss_val, op = session.run([loss, optimizer], feed_dict)
			#if epoch % 100 == 0:
			#	lr = lr/2
			#	print("new learning rate:", lr)
				print(loss_val)
			
			
			loss_val = session.run(loss, feed_dict = {x: x_batch, y: y_batch})
			print("loss:", loss_val.mean())
			y_pred_batch = session.run(y_pred, {x: x_batch})
			compute_accuracy(y_pred_batch, y_batch)
			lr = float(input("new learning rate: "))
			#optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
			
			
		y_pred_batch = session.run(y_pred, {x: x_batch})
		compute_accuracy(y_pred_batch, y_batch)
	
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
	
	model = model_shuffleNet_cifar10
	#analyze_model(model)
	run(model)
	#hej
	

	

