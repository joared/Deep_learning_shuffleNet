import tensorflow as tf

import numpy as np
	
def bn_relu(l):
	l = tf.layers.batch_normalization(l)
	l = tf.nn.relu(l)
	return l
	
def depthwise_conv(name, 
				x, 
				out_channel, 
				kernel_shape, 
				padding='SAME', 
				strides=1, 
				activation=tf.identity):
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
	in_shape = l.get_shape().as_list()
	in_channel = in_shape[-1]
	assert in_channel % group == 0, in_channel
	new_shape = [-1] + in_shape[-3:-1] + [group, in_channel // group]
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
	
def shufflenet_unit(name, l, out_channel, group, strides, shuffle=True):
	name = name + "_sh" if name else "sh"
	in_shape = l.get_shape().as_list()
	in_channel = in_shape[-1]
	shortcut = l

	# "We do not apply group convolution on the first pointwise layer
	#  because the number of input channels is relatively small."
	group = group if in_channel > 24 else 1
	l = pointwise_gconv(name + "_pw_1", l, out_channel // 4, group, bn_relu)
	
	if shuffle: l = channel_shuffle(l, group)
	l = depthwise_conv(name, l, out_channel // 4, 3, strides=strides)
	l = tf.layers.batch_normalization(l)

	# doubles outputs if strides = 1
	out_channel = out_channel if strides == 1 else out_channel - in_channel
	l = pointwise_gconv(name + "_pw_2", l, out_channel, group, tf.layers.batch_normalization)
	
	if strides == 1:	# unit (b)
		output = tf.nn.relu(shortcut + l)
	else:	# unit (c)
		shortcut = tf.nn.avg_pool(shortcut, [1,3,3,1], [1,2,2,1], padding='SAME')
		
		# which one to use?
		#output = tf.concat([shortcut, tf.nn.relu(l)], axis=3)
		output = tf.nn.relu(tf.concat([shortcut, l], axis=3))
	return output
	
def shufflenet_unit_v2(name, l, out_channel, group, strides, shuffle=True):
	name = name + "_sh" if name else "sh"
	in_shape = l.get_shape().as_list()
	in_channel = in_shape[-1]
	shortcut = l

	# "We do not apply group convolution on the first pointwise layer
	#  because the number of input channels is relatively small."
	group = group if in_channel > 24 else 1
	l = pointwise_gconv(name + "_pw_1", l, out_channel // 4, group, bn_relu)
	
	if shuffle: l = channel_shuffle(l, group)
	l = depthwise_conv(name, l, out_channel // 4, 3, strides=strides)
	l = tf.layers.batch_normalization(l)

	if strides == 1:	# unit (b)
		if out_channel == in_channel:
		# doubles outputs if strides = 1
		
			l = pointwise_gconv(name + "_pw_2", l, out_channel, group, tf.layers.batch_normalization)
			output = tf.nn.relu(shortcut + l)
		else:
			out_pw = out_channel - in_channel
			l = pointwise_gconv(name + "_pw_2", l, out_pw, group, tf.layers.batch_normalization)
			output = tf.nn.relu(tf.concat([shortcut, l], axis=3))
		
	else:	# unit (c)
		shortcut = tf.nn.avg_pool(shortcut, [1,3,3,1], [1,2,2,1], padding='SAME')
		
		# which one to use?
		#output = tf.concat([shortcut, tf.nn.relu(l)], axis=3)
		output = tf.nn.relu(tf.concat([shortcut, l], axis=3))
	return output 
	
def shufflenet_stage(stage_name, l, out_channel, repeat, group, shuffle=True):
	for i in range(repeat+1):
		name = '{}block{}'.format(stage_name, i)
		l = shufflenet_unit(name, l, out_channel, group, 2 if i == 0 else 1, shuffle)
	return l

if __name__ == "__main__":
	pass
	
	


