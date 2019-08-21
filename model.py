import os
import time
import random
import tensorflow as tf
import numpy as np
import pickle

#from utils import train_val_data_split
import utils
import layers

class Model(object):
	def __init__(self, name, model_func=None, inp_shape=None, out_shape=None):
		self.name = name
		self.model_func = model_func
		self.inp_shape = inp_shape
		self.out_shape = out_shape
		self._is_built = False
		self.sess = None
		self.saver = None
		
	def _set_attr_from_graph(self, sess):
		"""
		Set attributes given a session. 
		The attributes are assumed to have been added to the graph in build or load.
		"""
		self.sess = sess
		model_name = self.name
		graph = self.sess.graph
		self.x = graph.get_tensor_by_name("{}/x:0".format(model_name))
		self.y = graph.get_tensor_by_name("{}/y:0".format(model_name))
		self.pre_process = graph.get_tensor_by_name("{}/pre_process:0".format(model_name))
		self.y_pred = graph.get_tensor_by_name("{}/y_pred:0".format(model_name))
		self.loss = graph.get_tensor_by_name("{}/loss:0".format(model_name))
		self.cost = graph.get_tensor_by_name("{}/cost:0".format(model_name))
		self.optimizer = graph.get_operation_by_name("{}/optimizer".format(model_name))
		self.learning_rate = graph.get_tensor_by_name("{}/learning_rate:0".format(model_name))
		self.global_step = graph.get_tensor_by_name("{}/global_step:0".format(model_name))
		self.beta = graph.get_tensor_by_name("{}/beta:0".format(model_name))
		self.correct_pred = graph.get_tensor_by_name("{}/correct_pred:0".format(model_name))
		self.accuracy = graph.get_tensor_by_name("{}/accuracy:0".format(model_name))

	def flops(self):
		g = tf.get_default_graph()
		#g = self.sess.graph
		run_meta = tf.RunMetadata()
		opts = tf.profiler.ProfileOptionBuilder.float_operation()
		flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
		if flops is not None:
			print('TF stats gives',flops.total_float_ops)

	def build(self, loss=tf.losses.softmax_cross_entropy, optimizer=tf.train.GradientDescentOptimizer, lr=0.08, lr_min=0.02, lr_reduction_per_step=0, beta=0.0):
		"""
		
		Args:
			loss: loss function taking logits as input
			optimizer: tensorflow optimizer to be used during training	
			beta: regularization factor
		"""
		print("Model '{}' build:".format(self.name))
		print("Loss: {}".format(loss))
		print("Optimizer: {}".format(optimizer))
		print("Learning rate: {}, Min: {}, Red/step: {}".format(lr, lr_min, lr_reduction_per_step))
		print("Beta: {}".format(beta))
		if self._is_built is True: raise Exception("Model has allready been built, use clear to revert build and then build again")
		
		tf.reset_default_graph()
		sess = tf.Session()
		with tf.variable_scope(self.name):
			x = tf.placeholder(tf.float32, shape=self.inp_shape, name="x")
			y = tf.placeholder(tf.float32, shape=self.out_shape, name="y")
			pre_process = tf.image.per_image_standardization(x)
			#pre_process = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
			pre_process = tf.identity(pre_process, name="pre_process")
			
			# build hidden layers by calling model_func
			logits = self.model_func(x)
				
			if logits.get_shape().as_list() != list(self.out_shape): 
				raise Exception("Last layer has the wrong output shape. "\
								"{} != {}".format(logits.get_shape().as_list(), self.out_shape))
			
			y_pred = tf.nn.softmax(logits)
			y_pred = tf.identity(y_pred, name="y_pred")
			
			loss = loss(y, logits)
			loss = tf.identity(loss, name="loss")
			beta = tf.Variable(beta, trainable=False, name="beta")
			cost = loss
			for w in tf.trainable_variables():
				cost += beta*tf.nn.l2_loss(w)
			cost = tf.identity(cost, name="cost")
			global_step = tf.Variable(0, trainable=False, name="global_step")
			
			learning_rate = tf.maximum(lr - tf.dtypes.cast(global_step, tf.float32)*lr_reduction_per_step, lr_min)
			learning_rate = tf.placeholder_with_default(learning_rate, shape=[], name="learning_rate")
			
			optimizer = optimizer(learning_rate)
			optimizer = optimizer.minimize(cost, global_step, name="optimizer")
			correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1), name='correct_pred')
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
		
		sess.run(tf.global_variables_initializer())
		self._set_attr_from_graph(sess)
		self._is_built = True

	def load(self, path_to_models="models"):
		"""Load and build model from last checkpoint"""
		meta_name = self.name + ".meta"
		path = os.path.join(path_to_models, self.name, meta_name)
		
		tf.reset_default_graph()
		sess = tf.Session()
		
		# redundant
		assert len(sess.graph.get_operations()) == 0, "graph is not empty, "\
													"it has to be empty before loading a model"

		saver = tf.train.import_meta_graph(path)
		ckpt = tf.train.latest_checkpoint(os.path.join(path_to_models, self.name))

		graph = tf.get_default_graph()
		saver.restore(sess, ckpt)
		self._set_attr_from_graph(sess)
		self._is_built = True

	def save(self, path_to_models="models"):
		print(len(self.sess.graph.get_operations()))
		saver = tf.train.Saver()
		print(len(self.sess.graph.get_operations()))
		
		model_name = self.name
		if not os.path.isdir(path_to_models):
			os.mkdir(path_to_models)
		if not os.path.isdir(os.path.join(path_to_models, model_name)):
			os.mkdir(os.path.join(path_to_models, model_name))
		if self.sess._closed == False:
			path = os.path.join(path_to_models, model_name, model_name)
			global_step = self.sess.graph.get_tensor_by_name("{}/global_step:0".format(model_name))
			saver.save(self.sess, path) # save model
			saver.save(self.sess, path, global_step=global_step) # save model with global step
			print("Saved model!")
		else:
			raise Exception("Session is closed and save is not possible...")

	def exp_learning_rate(self, X, Y, lr_start, lr_end, n_iterations, batch_size=100):
		def moving_average(a, n=3) :
			ret = np.cumsum(a, dtype=float)
			ret[n:] = ret[n:] - ret[:-n]
			return ret[n - 1:] / n
		if not self._is_built: raise Exception("Model not built yet, use load or build to build model")
		with self.sess:
			
			# make pre processing a part of loading the batches instead
			print("preprocessing images...")
			X = self.sess.run(self.pre_process, {self.x: X})
			print("...preprocessing done!")
			
			# record training data
			losses = []
			learning_rates = []
			
			lr = lr_start
			c = pow(lr_end/lr_start, 1/n_iterations) # constant for increasing learning rate
			N = X.shape[0]
			perm = np.random.permutation(N)
			for i in range(n_iterations):
				
				#self.sess.run(tf.global_variables_initializer()) # trying remove if not working
				
				batch_inds = perm[i*batch_size:batch_size*(i+1)]
				feed_dict = {self.x: X[batch_inds], self.y: Y[batch_inds],
							self.learning_rate: lr}
				
				feed_dict = {self.x: X[0:batch_size], self.y: Y[0:batch_size],
							self.learning_rate: lr}
				
				# performs gradient descent and updates weights
				_, train_loss, train_cost, train_acc = self.sess.run([self.optimizer, 
																	self.loss, 
																	self.cost, 
																	self.accuracy], 
																	feed_dict)

				losses.append(train_loss)
				print(train_loss)
				learning_rates.append(lr)
				lr = lr_start * pow(c, i)
			
				print("learning rate:", self.learning_rate.eval({self.learning_rate: lr}))
				if train_loss > 100:
					break
			
		losses = moving_average(losses, n=3)
		return losses, learning_rates[0:len(losses)]

	def train(self, X, Y, batch_size=100, epochs=1, train_val_split=0.0, validation_data=None, validation_freq=1, save_data=True):
		#initial_epoch=0
		if not self._is_built: raise Exception("Model not built yet, use load or build to build model")
		if validation_data:
			X_val, Y_val = validation_data
		elif train_val_split:
			X, Y, X_val, Y_val = utils.train_val_data_split(X, Y, train_val_split)
		else:
			X_val, Y_val = None, None
		
		with self.sess:
			# make pre processing a part of loading the batches instead
			print("preprocessing images...")
			X = self.sess.run(self.pre_process, {self.x: X})
			if X_val is not None: X_val = self.sess.run(self.pre_process, {self.x: X_val})
			print("...preprocessing done!")
			
			# record training data
			losses = {"train": [], "validation": [], "test": []}
			costs = {"train": [], "validation": [], "test": []}
			accs = {"train": [], "validation": [], "test": []}
			N = X.shape[0]
			print("Training...")
			for epoch in range(1, epochs+1):
				perm = np.random.permutation(N)
				print("=========================")
				print("epoch", epoch, "(steps={})".format(self.global_step.eval()))
				
				# temporary lists of data to be saved later
				#train_losses = []
				#train_costs = []
				#train_accs = []
				
				batch_runs = int(N/batch_size)
				for i in range(batch_runs):
					# for estimating epoch time
					if i == 0 and epoch == 1: start_time = time.time()
					batch_inds = perm[i*batch_size:batch_size*(i+1)]
					feed_dict = {self.x: X[batch_inds], self.y: Y[batch_inds]}

					# performs gradient descent and updates weights
					_, train_loss, train_cost, train_acc = self.sess.run([self.optimizer, 
																		self.loss, 
																		self.cost, 
																		self.accuracy]
																		, feed_dict) 
					#train_losses.append(train_loss)
					#train_costs.append(train_cost)
					#train_accs.append(train_acc)
					
					losses["train"].append(train_loss)
					costs["train"].append(train_cost)
					accs["train"].append(train_acc)
					
					if i == 0 and epoch == 1:
						# estimate epoch time
						est_time = time.time()-start_time
						est_time *= (batch_runs-1)
						print(" - estimated epoch time (min):", time.strftime("%H:%M:%S",time.gmtime(est_time)))
					
					# linear learning rate decay
					#lr = lr-(lr_start-lr_end)/epochs
					#lr = exponential_lr(self.global_step.eval())
					#print("new learning rate:", round(lr,2))
				
				from_ind = (epoch-1)*batch_runs
				to_ind = epoch*batch_runs
				print("train loss:", np.mean(losses["train"][from_ind:to_ind]))
				print("train cost:", np.mean(costs["train"][from_ind:to_ind]))
				print("train acc:", np.mean(accs["train"][from_ind:to_ind]))
				
				if epoch % validation_freq == 0 and X_val is not None and Y_val is not None:
					# perform validation every validation_freq epoch
					print("Validating...")
					val_y_pred, val_loss, val_cost, val_acc = self.sess.run([self.y_pred, 
																			self.loss, 
																			self.cost, 
																			self.accuracy], 
																			feed_dict = {self.x: X_val, self.y: Y_val})
					losses["validation"].append(val_loss)
					costs["validation"].append(val_cost)
					accs["validation"].append(val_acc)
					print("val loss:", val_loss)
					print("val cost:", val_cost)
					print("val acc:", val_acc)
				
				# linear learning rate decay
				#lr = lr-(lr_start-lr_end)/epochs
				#print("new learning rate:", round(lr,2))
				print("learning rate:", self.learning_rate.eval())
			print("... done!")
			if save_data:
				utils.save_training_data([losses, costs, accs], self.name)
				self.save()
			
		return losses, costs, accs
	

	def evaluate(self):
		pass
			
	def evaluate_prediction_time(self, X, n_predictions=10000):
		print("Evaluating average prediction time with {} predictions".format(n_predictions))
		assert self._is_built, "Model not built and can not be evaluated"
		t_start = time.time()
		for i in range(n_predictions):
			ind = random.randint(0, X.shape[0])
			pred = self.sess.run(self.y_pred, {self.x: X[ind:ind+1]})
		
		t = time.time()-t_start
		average_pred_time = t/n_predictions
		average_pred_time *= 1000
		print("Average prediction time (ms):", average_pred_time)
		return average_pred_time
		
def cifar10_model(model_func):
	inp_shape=(None,32,32,3)
	out_shape=(None,10)
	return Model(model_func.__name__, model_func, inp_shape, out_shape)
	
def tiny200_model(model_func):
	inp_shape=(None,224,224,3)
	out_shape=(None,200)
	return Model(model_func.__name__, model_func, inp_shape, out_shape)
	
#shufflenet_model = default_model(shuffle_net)

if __name__ == "__main__":
	# TODO
	# Its only possible to train once since session is closed
	# therefore not possible to save after training
	
	#X = X[0:200, :, :, :]
	#Y = Y[0:200]
	#X_val = X_val[0:100, :, :, :]
	#Y_val = Y_val[0:100]	
	#X = np.concatenate([X, X_val], axis=0)
	#Y = np.concatenate([Y, Y_val], axis=0)
	
	tf.logging.set_verbosity(tf.logging.ERROR)
	model = Model("test_1")
	#model.add(layers.DepthwiseConv(24, 3))
	#model.add(tf.layers.flatten)
	#model.add(tf.layers.dense, 10)
	#model.build()
	#model.load()
	X, Y = utils.load_dataset("cifar10")
	
	model.evaluate_prediction_time(X)
	model.train(X[:200, :, :, :], Y[:200], epochs=3)

	#model.add(shufflenet_unit)
	#model.add(shufflenet_model, group=3, shuffle=False)
	#model.add(DepthwiseConv(24, 3))
	#model.add(DepthwiseConv(24, 3))
	#model.add(tf.layers.flatten)
	#model.add(tf.layers.dense, 10)
	#model.build(lr_reduction_per_step=0.01)

	#X, Y = load_dataset("cifar10")
	#model.train(X[:200, :, :, :], Y[:200], epochs=3, )
	#model.summary()

