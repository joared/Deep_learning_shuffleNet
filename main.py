import os
import sys
import argparse
import tensorflow as tf

import time
import numpy as np

import pickle
import matplotlib.pyplot as plt

from utils import load_dataset, save_training_data, load_training_data, plot_training_data, plot_image, plot_exp_learning_rate
import utils
import inspect
import model_builds
import model
#from models import load_model
#from model_builder import shufflenet_model_cifar10_small

def conv_flops(inp_dim, inp_chan, kernel, filters, stride):
	return pow(inp_dim, 2)*pow(kernel, 2)*inp_chan*filters/pow(stride, 2)

def shuffle_unit_flops(inp_size, inp_chan, out_chan, group, bottleneck_div=4):
	bottleneck_chan = out_chan//bottleneck_div
	return pow(inp_size, 2)*bottleneck_chan*(2*inp_chan/group + 9)

def shuffle_stage_flops(inp_size, inp_chan, out_chan, repeat, group, bottleneck_div=4):
	s = 0
	for i in range(repeat+1):
		if i == 0 and inp_chan <= 24:
			b_neck = out_chan//bottleneck_div
			s += pow(inp_size, 2)*(inp_chan*b_neck + 1/4*b_neck*(4*b_neck-inp_chan)/group + 9/4*b_neck)
		elif i == 0:
			s += pow(inp_size, 2)*inp_chan*(3/2*inp_chan/group + 9/8)
		else:
			s += shuffle_unit_flops(inp_size, out_chan, out_chan, group, bottleneck_div)
	return s

def shuffle_stage_flops_old(inp_size, inp_chan, out_chan, repeat, group):
	s = 0
	for i in range(repeat+1):
		if i == 0:
			s += pow(inp_size, 2)*inp_chan*(3/2*inp_chan/group + 9/8)
		else:
			s += shuffle_unit_flops(inp_size, out_chan, out_chan, group)
	return s
	
def choose_model(args):
	models = []
	i = 1
	print("Choose model: ")
	for mem in inspect.getmembers(model_builds):
		if isinstance(mem[-1], model.Model):
			models.append(mem[-1])
			print("{}. {}".format(i, mem[-1].name))
			i += 1
		
	choice = int(input("Choose: "))
	m = models[choice-1]
	tf.logging.set_verbosity(tf.logging.ERROR)

	m.build(lr=args.lr, lr_min=args.lr_min, lr_reduction_per_step=args.lr_red, beta=args.beta)
	#m.flops()
	weights = 0
	for i in tf.trainable_variables():
		p = 1
		for j in i.shape:
			p *= j
		weights += p
	print("Weights: {}".format(weights))
	
	return m
	
def eval_pred_time(args):
	m = choose_model(args)
	X, Y = load_dataset("cifar10")
	pred_time = m.evaluate_prediction_time(X, args.pred_iterations)
	
def train_plot(args):
	m = choose_model(args)
	d = utils.load_training_data(m.name)
	utils.plot_training_data(d[0], d[1], d[2])
	
def lr_test(args):
	m = choose_model(args)
	X, Y = load_dataset("cifar10")
	losses, learning_rates = m.exp_learning_rate(X, Y, lr_start=0.00001, lr_end=1, n_iterations=args.lr_iterations, batch_size=100)
	lr_test = {"losses": losses, "learning rates": learning_rates}
	utils.save_any("lr_test", lr_test)
	plot_exp_learning_rate(losses, learning_rates)
	
def train(args):
	m = choose_model(args)
	X, Y = load_dataset("cifar10")
	m.train(X, Y, batch_size=100, epochs=args.epochs, train_val_split=args.data_split, validation_freq=args.val_freq, save_data=True)
	
def main(args):
	#m.evaluate_prediction_time(X, n_predictions=100)
	#m.train(X[:200, :, :, :], Y[:200], batch_size=10, epochs=10, save_data=False)
	
	choices = [train, lr_test, train_plot, eval_pred_time]
	for i, c in enumerate(choices):
		print("{}. {}".format(i+1, c.__name__))
	choice = int(input(": "))
	choices[choice-1](args)

	
if __name__ == "__main__":
	# conv_flops(inp_dim, inp_chan, kernel, filters, stride)
	# shuffle_unit_flops(inp_size, inp_chan, out_chan, group)
	# shuffle_stage_flops(inp_size, inp_chan, out_chan, repeat, group)
	
	### TODO ###
	# import pictures as train, val and test
	# run from terminal
	# concat training data saves
	
	parser = argparse.ArgumentParser()
	# training variables
	parser.add_argument('--epochs', type=int, default=5, help='epochs')
	parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument('--lr_min', type=float, default=0.00001, help='learning rate minimum value')
	parser.add_argument('--lr_red', type=float, default=0.0, help='learning rate reduction per update')
	parser.add_argument('--beta', type=float, default=0.0, help='beta')
	parser.add_argument('--data_split', type=float, default=0.0, help='train/val data split')
	
	# test variables
	parser.add_argument('--lr_iterations', type=int, default=100, help='lr test iterations')
	parser.add_argument('--pred_iterations', type=int, default=10000, help='prediction time iterations')
	
	#parser.add_argument('--data', default="cifar10", help='dataset')
	#parser.add_argument('--batch', type=int, default=100, help='batch size')
	#parser.add_argument('--load', help='model name')
	#parser.add_argument('--shuffle', type=int, default=1, help='shuffle')
	##parser.add_argument('--eval', action='store_true')
	##parser.add_argument('--flops', action='store_true', help='print flops and exit')
	
	args = parser.parse_args()
	main(args)
	#test(args)
	


	

