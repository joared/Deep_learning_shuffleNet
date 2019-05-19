import os
import sys
import argparse
import tensorflow as tf

import time
import numpy as np

import pickle
import matplotlib.pyplot as plt

from shufflenet_utils import load_dataset, save_training_data, load_training_data, plot_training_data
from shufflenet_models import load_model
from shufflenet_model_builder import shufflenet_model_cifar10_small

def exponential_lr(step):
	return 0.000001*pow(10, step/65)

def train(model, epochs, lr, batch_size, dataset="cifar10"):
	"""
	Trains model with training data from data_path.
	"""
	X, Y, X_val, Y_val = load_dataset(dataset)

	#X = X[0:200, :, :, :]
	#Y = Y[0:200]
	#X_val = X_val[0:100, :, :, :]
	#Y_val = Y_val[0:100]
	
	#X = np.concatenate([X, X_val], axis=0)
	#Y = np.concatenate([Y, Y_val], axis=0)
	
	N = X.shape[0]
	
	model_name = model.name
	session = model.sess
	x = model.x
	y = model.y
	y_pred = model.y_pred
	loss = model.loss
	cost = model.cost
	optimizer = model.optimizer
	learning_rate = model.learning_rate
	lr_start = lr
	lr_end = 0.02
	lr = lr_start
	
	with session:
		print("preprocessing images...")
		X = session.run(model.pre_process, {x: X})
		X_val = session.run(model.pre_process, {x: X_val})
		print("...preprocessing done!")
		
		# record training data
		losses = {"train": [], "validation": [], "test": []}
		accs = {"train": [], "validation": [], "test": []}
		
		print("Training...")
		for epoch in range(1, epochs+1):
			perm = np.random.permutation(N)
			print("=========================")
			print("epoch", epoch, "(steps={})".format(model.global_step.eval()))
			
			batch_runs = int(N/batch_size)
			for i in range(batch_runs):
				if i == 0 and epoch == 1: start_time = time.time()
				inds = perm[i*batch_size:batch_size*(i+1)]
				feed_dict = {x: X[inds], y: Y[inds], learning_rate: lr}
				_, train_loss = session.run([optimizer, loss], feed_dict) # performs gradient descent
				#losses["train"].append(train_loss)
				
				if i == 0 and epoch == 1:
					est_time = time.time()-start_time
					est_time *= (batch_runs-1)
					print(" - estimated epoch time (min):", time.strftime("%H:%M:%S",time.gmtime(est_time)))
				
				# linear learning rate decay
				#lr = lr-(lr_start-lr_end)/epochs
				#lr = exponential_lr(model.global_step.eval())
				#print("new learning rate:", round(lr,2))
					
			train_y_pred, train_loss, train_cost = session.run([y_pred, loss, cost], feed_dict = {x: X, y: Y})
			train_acc = model.compute_accuracy(train_y_pred, Y)
			losses["train"].append(train_loss)
			accs["train"].append(train_acc)
			
			val_y_pred, val_loss, val_cost = session.run([y_pred, loss, cost], feed_dict = {x: X_val, y: Y_val})
			val_acc = model.compute_accuracy(val_y_pred, Y_val)
			losses["validation"].append(val_loss)
			accs["validation"].append(val_acc)
			
			print("train loss:", train_loss)
			print("train cost:", train_cost.mean())
			print("train acc:", train_acc)
			print("val loss:", val_loss.mean())
			print("val cost:", val_cost.mean())
			print("val acc:", val_acc)
			
			# linear learning rate decay
			lr = lr-(lr_start-lr_end)/epochs
			print("new learning rate:", round(lr,2))
		
		save_training_data([losses, accs], model_name)
		model.save()
		
	return losses, accs
	

def get_good_learning_rate(lr_start, batch_size=100):
	model = shufflenet_model_cifar10_big("learning_rate")
	losses, accs = train(model, epochs=4, lr=lr_start, batch_size=100, dataset="cifar10")
	plot_training_data(losses, None)
	
def test2():
	#data = load_dataset("tiny200")
	#print(data)
	model_name = "learning_rate"
	losses, accs = load_training_data(model_name)
	#model_name_2 = "test_model_conv"
	plot_training_data(losses, accs, model_name)
	input()
	quit()
	
def test(args):
	# test to see if run if working properly
	
	model_name = args.model_name
	
	if args.load:
		model = load_model(model_name)
	else:
		model = shufflenet_model_cifar10_small(model_name, args.group, bool(args.shuffle))
	model.sess.run(model.beta.assign(args.beta))
	
	losses, accs = train(model, args.epochs, args.lr, args.batch)
	print("================")
	print("losses:", len(losses["train"]))
	print("accs:", len(accs["train"]))
	print("Test finnished")
	input()
	quit()
		
if __name__ == "__main__":
	### TODO ###
	# import pictures as train, val and test
	# run from terminal
	# concat training data saves
	#test2()
	parser = argparse.ArgumentParser()
	parser.add_argument('model_name', help='mode name')
	parser.add_argument('--epochs', type=int, default=5, help='epochs')
	parser.add_argument('--lr', type=float, default=0.083, help='learning rate')
	
	parser.add_argument('--data', default="cifar10", help='dataset')
	parser.add_argument('--batch', type=int, default=100, help='batch size')
	parser.add_argument('--load', help='model name')
	parser.add_argument('--beta', type=float, default=0, help='beta')
	parser.add_argument('--group', type=int, default=1, help='group')
	parser.add_argument('--shuffle', type=int, default=1, help='shuffle')
	#parser.add_argument('--eval', action='store_true')
	
	#parser.add_argument('--flops', action='store_true', help='print flops and exit')
	args = parser.parse_args()
	test(args)


	

