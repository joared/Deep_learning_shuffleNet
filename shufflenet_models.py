import os
import tensorflow as tf
import numpy as np

from shufflenet_layers import *

class Model:
	def __init__(self, 
				name,
				x,
				y,
				pre_process,
				y_pred, 
				loss, 
				optimizer, 
				learning_rate,
				global_step,
				session=None):
		
		assert isinstance(name, str), name
		if session is not None: assert isinstance(session, tf.Session)
		
		for var, var_name in zip([x,y,pre_process,y_pred,loss,optimizer,learning_rate,global_step],
							["x:0","y:0","pre_process:0","y_pred:0","loss:0","optimizer","learning_rate:0","global_step:0"]):
			var_name = "{}/{}".format(name, var_name)
			if var is not None: assert var.name == var_name, "{} != {}".format(var.name, var_name)
		
		self.name = name
		self.sess = session
		self.x = x
		self.y = y
		self.pre_process = pre_process
		self.y_pred = y_pred
		self.loss = loss
		self.optimizer = optimizer 
		self.learning_rate = learning_rate
		self.global_step = global_step
		
		if session is None:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())

	def compute_accuracy(self, y_pred_value, Y):
		s = 0
		for p, y in zip(y_pred_value, Y):
			p_index = np.argmax(p)
			y_index = np.argmax(y)
			if p_index == y_index:
				s += 1
				
		acc = s/y_pred_value.shape[0]
		
		return acc

	def save(self):
		saver = tf.train.Saver() # for saving model
		model_name = self.name
		if not os.path.isdir("models"):
			os.mkdir("models")
		if not os.path.isdir(os.path.join("models", model_name)):
			os.mkdir(os.path.join("models", model_name))
		if self.sess._closed == False:
			path = os.path.join("models", model_name, model_name)
			global_step = self.sess.graph.get_tensor_by_name("{}/global_step:0".format(model_name))
			saver.save(self.sess, path) # save model
			saver.save(self.sess, path, global_step=global_step) # save model
		else:
			print("Session is closed and save is not possible...")
			
	def load(model_name):
		meta_name = model_name + ".meta"
		#if step is not None:
		#	meta_name = model_name + "-" + str(step) + ".meta"
		path = os.path.join("models", model_name, meta_name)
		
		sess = tf.Session()
		assert len(sess.graph.get_operations()) == 0, "graph is not empty, "\
													"it has to be empty before loading a model"

		saver = tf.train.import_meta_graph(path)
		ckpt = tf.train.latest_checkpoint(os.path.join("models", model_name))

		graph = tf.get_default_graph()
		saver.restore(sess, ckpt)
		
		x = graph.get_tensor_by_name("{}/x:0".format(model_name))
		y = graph.get_tensor_by_name("{}/y:0".format(model_name))
		pre_process = graph.get_tensor_by_name("{}/pre_process:0".format(model_name))
		y_pred = graph.get_tensor_by_name("{}/y_pred:0".format(model_name))
		loss = graph.get_tensor_by_name("{}/loss:0".format(model_name))
		optimizer = graph.get_operation_by_name("{}/optimizer".format(model_name))
		learning_rate = graph.get_tensor_by_name("{}/learning_rate:0".format(model_name))
		try:
			global_step = graph.get_tensor_by_name("{}/global_step:0".format(model_name))
		except:
			print("no global step tensor found...")
		
		return Model(model_name, x, y, pre_process, y_pred, loss, optimizer, learning_rate, global_step, session=sess)

def default_model(model_func):
	# Change this wrapper, i doesnt make sense
	def wrap(name, *args, **kwargs):
		with tf.variable_scope(name):
			x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name="x")
			y = tf.placeholder(tf.float32, shape=(None, 10), name="y")
			pre_process = tf.image.per_image_standardization(x)
			pre_process = tf.identity(pre_process, name="pre_process")
			
			layer = model_func(name, x, *args, *kwargs)
			y_pred = tf.identity(layer, name="y_pred")
			
			loss = tf.losses.softmax_cross_entropy(y, y_pred)
			loss = tf.identity(loss, name="loss")
			learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
			global_step = tf.Variable(0, trainable=False, name="global_step")
			optimizer = tf.train.GradientDescentOptimizer(learning_rate)
			optimizer = optimizer.minimize(loss, global_step, name="optimizer")
			
		return Model(name, x, y, pre_process, y_pred, loss, optimizer, learning_rate, global_step)
	return wrap

load_model = Model.load

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
	
	init = tf.global_variables_initializer()
	with tf.Session() as session:
		writer = tf.summary.FileWriter("C:\\Users\\Joris\\deep_learning\\graphs", session.graph)
		session.run(init)
		test_val = session.run(test, {x: x_data})
		
	print(test_val.shape)
	print(test_val)
	
	


