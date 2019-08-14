import os
import sys
import argparse
import tensorflow as tf

import time
import numpy as np

import pickle
import matplotlib.pyplot as plt

from utils import load_dataset, save_training_data, load_training_data, plot_training_data, plot_image
#from models import load_model
#from model_builder import shufflenet_model_cifar10_small

def conv_flops(inp_dim, inp_chan, kernel, filters, stride):
	return pow(inp_dim, 2)*pow(kernel, 2)*inp_chan*filters/pow(stride, 2)

def shuffle_unit_flops(inp_size, inp_chan, out_chan, group):
	bottleneck_chan = out_chan//4
	return pow(inp_size, 2)*bottleneck_chan*(2*inp_chan/group + 9)

def shuffle_stage_flops(inp_size, inp_chan, out_chan, repeat, group):
	s = 0
	for i in range(repeat+1):
		if i == 0 and inp_chan <= 24:
			b_neck = out_chan//4
			s += pow(inp_size, 2)*(inp_chan*b_neck + 1/4*b_neck*(4*b_neck-inp_chan)/group + 9/4*b_neck)
		elif i == 0:
			s += pow(inp_size, 2)*inp_chan*(3/2*inp_chan/group + 9/8)
		else:
			s += shuffle_unit_flops(inp_size, out_chan, out_chan, group)
	return s

def shuffle_stage_flops_old(inp_size, inp_chan, out_chan, repeat, group):
	s = 0
	for i in range(repeat+1):
		if i == 0:
			s += pow(inp_size, 2)*inp_chan*(3/2*inp_chan/group + 9/8)
		else:
			s += shuffle_unit_flops(inp_size, out_chan, out_chan, group)
	return s

def get_good_learning_rate(lr_start, batch_size=100):
	model = shufflenet_model_cifar10_big("learning_rate")
	losses, accs = train(model, epochs=4, lr=lr_start, batch_size=100, dataset="cifar10")
	plot_training_data(losses, None)
	
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
		
def main(args):
	import inspect
	import model_builds
	import model
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
	#import model
	#m = model.Model("model_conv")
	#m.load()

	m.build(lr=args.lr, beta=args.beta)
	m.flops()
	#print("m1 ops:", len(m1.sess.graph.get_operations()))
	
	#m2.build()
	#print("m2 ops:", len(m2.sess.graph.get_operations()))
	
	X, Y = load_dataset("cifar10")
	
	#m.evaluate_prediction_time(X, n_predictions=100)
	#m.train(X[:200, :, :, :], Y[:200], batch_size=10, epochs=10, save_data=False)
	weights = 0
	for i in tf.trainable_variables():
		p = 1
		for j in i.shape:
			p *= j
		weights += p
	print("Weights: {}".format(weights))
	input()
	m.train(X, Y, batch_size=100, epochs=args.epochs, train_val_split=args.data_split, save_data=True)

if __name__ == "__main__":
	# conv_flops(inp_dim, inp_chan, kernel, filters, stride)
	# shuffle_unit_flops(inp_size, inp_chan, out_chan, group)
	# shuffle_stage_flops(inp_size, inp_chan, out_chan, repeat, group)
	
	### TODO ###
	# import pictures as train, val and test
	# run from terminal
	# concat training data saves
	
	parser = argparse.ArgumentParser()
	#parser.add_argument('model_name', help='mode name')
	parser.add_argument('--epochs', type=int, default=5, help='epochs')
	parser.add_argument('--lr', type=float, default=0.083, help='learning rate')
	
	#parser.add_argument('--data', default="cifar10", help='dataset')
	#parser.add_argument('--batch', type=int, default=100, help='batch size')
	#parser.add_argument('--load', help='model name')
	parser.add_argument('--beta', type=float, default=0.0, help='beta')
	#parser.add_argument('--group', type=int, default=1, help='group')
	#parser.add_argument('--shuffle', type=int, default=1, help='shuffle')
	##parser.add_argument('--eval', action='store_true')
	
	##parser.add_argument('--flops', action='store_true', help='print flops and exit')
	parser.add_argument('--data_split', type=float, default=0.0, help='train/val data split')
	args = parser.parse_args()
	main(args)
	#test(args)
	


	

