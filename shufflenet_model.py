import tensorflow as tf

import numpy as np
	
def bn_relu(l):
	l = tf.layers.batch_normalization(l)
	l = tf.nn.relu(l)
	return l
	
def depthwise_conv(name, x, out_channel, kernel_shape, padding='SAME', strides=1, activation=tf.identity):
	# not tested
	in_shape = x.get_shape().as_list()
	in_channel = in_shape[-1]
	assert out_channel % in_channel == 0, (out_channel, in_channel)
	channel_mult = out_channel // in_channel

	kernel_shape = [kernel_shape, kernel_shape]
	filter_shape = kernel_shape + [in_channel, channel_mult]
	W_init = tf.variance_scaling_initializer(2.0)
	W = tf.get_variable(name + '_dw', filter_shape, initializer=W_init)
	conv = tf.nn.depthwise_conv2d(x, W, [1, strides, strides, 1], padding=padding)
	return activation(conv)
	
	
def channel_shuffle(l, group):
	# tested and seems to work
	in_shape = l.get_shape().as_list()
	in_channel = in_shape[-1]
	#print("In channels", in_channel)
	assert in_channel % group == 0, in_channel
	new_shape = [-1] + in_shape[-3:-1] + [group, in_channel // group]
	#print("new shape", new_shape)
	l = tf.reshape(l, new_shape)
	l = tf.transpose(l, [0, 1, 2, 4, 3])
	l = tf.reshape(l, [-1] + in_shape[-3:-1] + [in_channel])
	return l	
	
def pointwise_gconv(name, l, out_channel, group, activation):
	name = name + "_pw_" if name else "pw_"
	in_shape = l.get_shape().as_list()
	in_channel = in_shape[-1]
	
	l_list = tf.split(l, num_or_size_splits=group, axis=3) # list of nodes
	
	filter_shape = [1, 1, in_channel//group, out_channel]
	W_init = tf.variance_scaling_initializer(2.0)
	W = tf.get_variable(name, filter_shape, initializer=W_init)
	kernels = tf.split(W, group, 3)
	l_list_2 = [];
	for t, k in zip(l_list, kernels):
		#apply conv on each group


		t = tf.nn.conv2d(t, k, [1,1,1,1], "SAME")
		l_list_2.append(t)
	
	l = tf.concat(l_list_2, 3)
	return activation(l)
	
def shufflenet_unit(name, l, out_channel, group, strides):
	name = name + "_sh" if name else "sh"
	in_shape = l.get_shape().as_list()
	in_channel = in_shape[-1]
	shortcut = l
	
	
	# "We do not apply group convolution on the first pointwise layer
	#  because the number of input channels is relatively small."
	first_split = group if in_channel > 24 else 1
	l = pointwise_gconv(name + "_pw_1", l, out_channel // 4, group, bn_relu)
	
	l = channel_shuffle(l, group)
	l = depthwise_conv(name, l, out_channel // 4, 3, strides=strides)
	l = tf.layers.batch_normalization(l)

	# doubles outputs if strides = 1
	chan_out = out_channel if strides == 1 else out_channel - in_channel
	
	l = pointwise_gconv(name + "_pw_2", l,
			out_channel if strides == 1 else out_channel - in_channel, 
			group, tf.layers.batch_normalization)
	l = tf.layers.batch_normalization(l)
	if strides == 1:     # unit (b)
		output = tf.nn.relu(shortcut + l)
	else:   # unit (c)
		shortcut = tf.nn.avg_pool(shortcut, [1,3,3,1], [1,2,2,1], padding='SAME')
		output = tf.concat([shortcut, tf.nn.relu(l)], axis=3)
		#output = tf.nn.relu(tf.concat([shortcut, l], axis=3))
	return output
	
def shufflenet_stage_1():
	pass
	
def model_shuffleNet_cifar10():
	input_image = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
	y = tf.placeholder(tf.float32, shape=(None, 10))
	print(input_image.shape)
	with tf.variable_scope("dafakh"):
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
		#l = tf.layers.batch_normalization(input_image)
		l = tf.image.per_image_standardization(input_image)
		l = tf.layers.conv2d(l, 24, 3, strides=2, padding="same")
		l = bn_relu(l)
		#l = tf.nn.max_pool(l, [1,3,3,1], [1,2,2,1], padding="SAME")
		
		# maxpooling
		#l = shufflenet_stage_1(l)
		group = 8
		channels = {1:[144, 288, 576],
					2:[200, 400, 800],
					3:[240, 480, 960],
					4:[272, 544, 1088],
					8:[384, 768, 1536]}
		l = shufflenet_unit("sh1", l, channels[group][0], group, 2)
		l = shufflenet_unit("sh2", l, channels[group][0], group, 1)
		l = shufflenet_unit("sh3", l, channels[group][0], group, 1)
		l = shufflenet_unit("sh4", l, channels[group][0], group, 1)
		
		"""
		l = shufflenet_unit("sh5", l, channels[group][1], group, 2)
		l = shufflenet_unit("sh6", l, channels[group][1], group, 1)
		l = shufflenet_unit("sh7", l, channels[group][1], group, 1)
		l = shufflenet_unit("sh8", l, channels[group][1], group, 1)
		l = shufflenet_unit("sh9", l, channels[group][1], group, 1)
		l = shufflenet_unit("sh10", l, channels[group][1], group, 1)
		l = shufflenet_unit("sh11", l, channels[group][1], group, 1)
		l = shufflenet_unit("sh12", l, channels[group][1], group, 1)
		"""
		#l = shufflenet_unit("sh13", l, channels[group][2], group, 2)
		#l = shufflenet_unit("sh14", l, channels[group][2], group, 1)
		#l = shufflenet_unit("sh15", l, channels[group][2], group, 1)
		#l = shufflenet_unit("sh16", l, channels[group][2], group, 1)
		
		
		#l = tf.reduce_mean(l, [1,2]) # global avg pooling
		#l = tf.nn.relu(l)
		l = tf.layers.flatten(l)
		#l = tf.layers.dense(l, 50, activation="relu")
		y_pred = tf.layers.dense(l, 10)#, activation = "relu")
		
		loss = tf.losses.softmax_cross_entropy(y, y_pred)
		#loss = tf.reduce_mean(tf.square(y_pred-y), 1)
		
	return input_image, y, y_pred, loss

def model_shuffleNet_imagenet():
	x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
	y = tf.placeholder(tf.float32, shape=(None, 200))
	print(x.shape)
	with tf.variable_scope("dafakh"):
		conv2d_1 = tf.layers.conv2d(x, 24, (3,3), strides=2, padding="same")
		conv2d_1 = tf.layers.batch_normalization(conv2d_1)
		conv2d_1 = tf.nn.relu(conv2d_1)
		
		conv2d_2 = tf.layers.conv2d(conv2d_1, 48, (2,2), strides=1, padding="same")
		conv2d_2 = tf.layers.batch_normalization(conv2d_2)
		conv2d_2 = tf.nn.relu(conv2d_2)
		
		h = tf.layers.dense(conv2d_2, 50, activation="relu")#, activation = "relu")
		h = tf.layers.flatten(h)
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

		session.run(init)
		
		y_pred_batch = session.run(y_pred, {x: x_batch})


if __name__ == "__main__":
	channels = 24
	size = 224
	x_data = np.zeros([1,size, size, channels])
	for i in range(channels):
		x_data[0,:,:,i] = np.ones([size,size])*i
		print(x_data[0,:,:,i])

	x = tf.placeholder(tf.float32, shape=(None, size, size, channels))
	y = tf.placeholder(tf.float32, shape=(None, 10))
	
	test = shufflenet_unit("s1", x, 144, 3, 2)
	#test = shufflenet_unit("shuffle_2", test, 144, 3, 1)
	#np.float32([3,3,8,1])
	#w = tf.Variable(np.float32(np.random.normal(size = [3, 3, 8, 2])))
	#test = tf.nn.depthwise_conv2d(x, w, [1,1,1,1], "VALID")
	#l = tf.layers.conv2d(x, 24, 1, strides=2, padding="same")
	#l = shufflenet_unit(l, 24, group=4, strides=1)
	#l = tf.layers.flatten(l)
	#y_pred = tf.layers.dense(l, 10)
	
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		writer = tf.summary.FileWriter("C:\\Users\\Joris\\deep_learning\\graphs", session.graph)
		session.run(init)
		test_val = session.run(test, {x: x_data})
		
	print(test_val.shape)
	print(test_val)
	
	


