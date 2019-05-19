import os
import tensorflow as tf
import numpy as np

from shufflenet_models import default_model
from shufflenet_layers import *

@default_model
def shufflenet_model_cifar10_big(name, input_image):
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
	
	l = tf.layers.conv2d(input_image, 24, 3, strides=1, padding="same")
	l = bn_relu(l)
	
	#l = tf.nn.max_pool(l, [1,3,3,1], [1,2,2,1], padding="SAME")

	group = 3
	channels = {1:[144, 288, 576],
				2:[200, 400, 800],
				3:[240, 480, 960],
				4:[272, 544, 1088],
				8:[384, 768, 1536]}
	l = shufflenet_stage("stage_1", l, channels[group][0], 3, group)
	l = shufflenet_stage("stage_2", l, channels[group][1], 7, group)
	l = shufflenet_stage("stage_3", l, channels[group][2], 3, group)
	
	# global avg pooling with relu
	#l = tf.reduce_mean(l, [1,2])
	#l = tf.nn.relu(l)
	l = tf.layers.flatten(l)
	
	#l = tf.layers.dense(l, 50, activation="relu")
	l = tf.layers.dense(l, 10, use_bias=True)
	return l

@default_model
def shufflenet_model_cifar10_small(name, input_image, group, shuffle):
	l = tf.layers.conv2d(input_image, 24, 3, strides=1, padding="same")
	l = bn_relu(l)
	
	channels = {1:[144, 288, 576],
				2:[200, 400, 800],
				3:[240, 480, 960],
				4:[272, 544, 1088],
				8:[384, 768, 1536]}
	#l = shufflenet_unit("sh1", l, channels[group][0], group, 2)
	l = shufflenet_unit_v2("sh2", l, channels[group][0], group, 1, shuffle)
	l = shufflenet_unit("sh3", l, channels[group][0], group, 1, shuffle)
	l = shufflenet_unit("sh4", l, channels[group][0], group, 1, shuffle)
	
	l = shufflenet_stage("stage_1", l, channels[group][1], 3, group, shuffle)
	
	l = tf.layers.flatten(l)
	l = tf.layers.dense(l, 10, use_bias=True)
	return l
	
@default_model
def model_conv(name, input_image):
	x_norm = tf.image.per_image_standardization(input_image)
	conv2d_1 = tf.layers.conv2d(input_image, 24, (3,3), strides=2, padding="same")
	conv2d_1 = tf.layers.batch_normalization(conv2d_1)
	conv2d_1 = tf.nn.relu(conv2d_1)
	
	conv2d_2 = tf.layers.conv2d(conv2d_1, 48, (2,2), strides=1, padding="same")
	conv2d_2 = tf.layers.batch_normalization(conv2d_2)
	conv2d_2 = tf.nn.relu(conv2d_2)
	
	h = tf.layers.flatten(conv2d_2)
	h = tf.layers.dense(h, 10, activation="relu")#, activation = "relu")
	#h = tf.layers.dense(h, 10, activation="relu")#, activation = "relu")
	return h

if __name__ == "__main__":
	pass
	


