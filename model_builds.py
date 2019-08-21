import os
import tensorflow as tf
import numpy as np

import utils
import model
from model import cifar10_model, tiny200_model
from layers import *

def model_wrapper(inp_shape, out_shape):
	def get_model(model_func):
		return model.Model(model_func.__name__, model_func, inp_shape, out_shape)
	return get_model
#def cifar10_model(model_func):#
#	inp_shape=(None,32,32,3)
	#out_shape=(None,10)
	#return Model(model_func.__name__, model_func, inp_shape, out_shape)
	
#def tiny200_model(model_func):
#	inp_shape=(None,224,224,3)
#	out_shape=(None,200)
#	return Model(model_func.__name__, model_func, inp_shape, out_shape)

channels = {1:[144, 288, 576],
			2:[200, 400, 800],
			3:[240, 480, 960],
			4:[272, 544, 1088],
			8:[384, 768, 1536]}

@model_wrapper(inp_shape=(None,224,224,3), out_shape=(None,1000))
def shufflenet_cifar10_original(input_image):
	"""
	l = Conv2D('conv1', image, first_chan, 3, strides=2, activation=BNReLU)
	l = MaxPooling('pool1', l, 3, 2, padding='SAME')

	l = shufflenet_stage('stage2', l, channels[0], 4, group)
	l = shufflenet_stage('stage3', l, channels[1], 8, group)
	l = shufflenet_stage('stage4', l, channels[2], 4, group)

	if args.v2:
		l = Conv2D('conv5', l, 1024, 1, activation=BNReLU)

	l = GlobalAvgPooling('gap', l)
	logits = FullyConnected('linear', l, 1000)
	"""
	group = 2
	channel_scale = 1
	
	l = tf.layers.conv2d(input_image, 24, 3, strides=2, padding="same")
	l = bn_relu(l)
	
	l = tf.nn.max_pool(l, [1,3,3,1], [1,2,2,1], padding="SAME")

	l = shufflenet_stage("stage_1", l, int(channel_scale*channels[group][0]), 3, group)
	l = shufflenet_stage("stage_2", l, int(channel_scale*channels[group][1]), 7, group)
	l = shufflenet_stage("stage_3", l, int(channel_scale*channels[group][2]), 3, group)
	
	# global avg pooling with relu
	print(l.shape)
	l = tf.reduce_mean(l, [1,2])
	print(l.shape)
	l = tf.nn.relu(l)
	
	l = tf.layers.flatten(l)
	#l = tf.layers.dense(l, 50, activation="relu")
	l = tf.layers.dense(l, 1000, use_bias=True)
	return l

@model_wrapper(inp_shape=(None,32,32,3), out_shape=(None,10))
def shufflenet_cifar10_x0_25(input_image):
	"""
	l = Conv2D('conv1', image, first_chan, 3, strides=2, activation=BNReLU)
	l = MaxPooling('pool1', l, 3, 2, padding='SAME')

	l = shufflenet_stage('stage2', l, channels[0], 4, group)
	l = shufflenet_stage('stage3', l, channels[1], 8, group)
	l = shufflenet_stage('stage4', l, channels[2], 4, group)

	if args.v2:
		l = Conv2D('conv5', l, 1024, 1, activation=BNReLU)

	l = GlobalAvgPooling('gap', l)
	logits = FullyConnected('linear', l, 1000)
	"""
	group = 3
	channel_scale = 0.25
	
	l = tf.layers.conv2d(input_image, int(channel_scale*24), 3, strides=1, padding="same")
	l = bn_relu(l)
	
	l = tf.nn.max_pool(l, [1,3,3,1], [1,2,2,1], padding="SAME")

	l = shufflenet_stage("stage_1", l, int(channel_scale*channels[group][0]), 3, group)
	l = shufflenet_stage("stage_2", l, int(channel_scale*channels[group][1]), 7, group)
	l = shufflenet_stage("stage_3", l, int(channel_scale*channels[group][2]), 3, group)
	
	# global avg pooling with relu
	print(l.shape)
	l = tf.reduce_mean(l, [1,2])
	print(l.shape)
	l = tf.nn.relu(l)
	
	l = tf.layers.flatten(l)
	#l = tf.layers.dense(l, 50, activation="relu")
	l = tf.layers.dense(l, 10, use_bias=True)
	return l

@cifar10_model
def shufflenet_cifar10_v1(input_image):
	# FLOPS: 12.9M
	# learning rate: 0.04
	group = 3
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 180, 3, group, shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l
	
@cifar10_model
def shufflenet_cifar10_v2(input_image):
	# FLOPS: 12.9M
	# learning rate: 0.04
	group = 3
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 60, 2, group, shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l
	
@cifar10_model
def shufflenet_cifar10_v3(input_image):
	# FLOPS: 1.06M
	# learning rate: 0.02
	group = 3
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 60, 1, group, shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l

@cifar10_model
def shufflenet_cifar10_v4(input_image):
	# FLOPS: 1.4M
	# learning rate: 0.02
	group = 3
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 72, 1, group, shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l

@cifar10_model
def shufflenet_cifar10_v5(input_image):
	# FLOPS: 1.4M
	# learning rate: 0.02
	group = 3
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 75, 1, group, shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l

@cifar10_model
def shufflenet_cifar10_v6(input_image):
	# FLOPS: 1.M
	# learning rate: 0.02
	group = 8
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 96, 1, group, shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l

@cifar10_model
def shufflenet_cifar10_v7(input_image):
	# FLOPS: 1.M
	# learning rate: 0.02
	group = 8
	shuffle = True
	l = input_image
	l = shufflenet_stage("stage_1", l, 64, 1, group, bottleneck_div=2, shuffle=shuffle)
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l

@cifar10_model
def conv_cifar10_v1(input_image):
	# FLOPS: 1 468 416
	conv2d_1 = tf.layers.conv2d(input_image, 24, (3,3), strides=2, padding="same")
	conv2d_1 = tf.layers.batch_normalization(conv2d_1)
	conv2d_1 = tf.nn.relu(conv2d_1)
	
	conv2d_2 = tf.layers.conv2d(conv2d_1, 48, (2,2), strides=1, padding="same")
	conv2d_2 = tf.layers.batch_normalization(conv2d_2)
	conv2d_2 = tf.nn.relu(conv2d_2)
	
	h = tf.layers.flatten(conv2d_2)
	#h = tf.layers.dense(h, 10, activation="relu")#, activation = "relu")
	h = tf.layers.dense(h, 10)#, activation = "relu")
	#h = tf.layers.dense(h, 10, activation="relu")#, activation = "relu")
	return h

# Jack
@cifar10_model
def conv_cifar10_jack(input_image):
	shuffle_group = 3
	# 100%, 50 epochs
	l = shufflenet_stage("stage_1", l=input_image, out_channel=120, repeat=2, group=shuffle_group)
	l = shufflenet_stage("stage_2", l=l, out_channel=240, repeat=4, group=shuffle_group)
	l = shufflenet_stage("stage_3", l=l, out_channel=480, repeat=2, group=shuffle_group)

	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10)
	return l


if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.ERROR)
	#import model
	#m1 = model.Model("model_conv")
	#m1.load()
	m1 = shufflenet_cifar10_v1
	m1 = conv_cifar10_v1
	m1.build()
	m1.flops()
	#print("m1 ops:", len(m1.sess.graph.get_operations()))
	
	#m2.build()
	#print("m2 ops:", len(m2.sess.graph.get_operations()))
	
	X, Y = utils.load_dataset("cifar10")
	#m1.evaluate_prediction_time(X, n_predictions=100)
	#m1.train(X[:200, :, :, :], Y[:200], batch_size=10, epochs=10, save_data=False)
	#m1.train(X, Y, batch_size=100, epochs=2, save_data=False)
	#print("m1 ops:", len(m1.sess.graph.get_operations()))

