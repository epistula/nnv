
import pdb
import numpy as np
import math
import scipy.misc
import platform
import matplotlib
from scipy.misc import imsave

if platform.dist()[0] == 'centos':
	matplotlib.use('Agg')
elif platform.dist()[0] == 'debian': 
	matplotlib.use('Agg')
elif platform.dist()[0] == 'Ubuntu': 
	print('On collab!')
else: 
	matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import scipy.stats as st
import seaborn as sns
import pickle
import zlib
import os
import uuid
import glob
import time
import shutil, errno
import tensorflow as tf
import string 
import pdb
import copy

plt.rcParams['axes.linewidth'] = 2

# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name=global_args.global_exp_dir+global_args.restore_dir+'/checkpoints/checkpoint', tensor_name='',all_tensors='')




class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old

def tf_print(input_tensor, list_of_print_tensors):
	input_tensor = tf.Print(input_tensor, list_of_print_tensors, message="Log values:")
	return input_tensor

def get_report_formatted(report, sess, curr_feed_dict):
    compute_list = []
    for e in report: 
        if e[2] is None: compute_list.append(e[1])
    computed_list = sess.run(compute_list, feed_dict = curr_feed_dict)

    report_value_list = []
    report_format = ''    
    curr_ind = 0
    for e in report: 
        if e[2] is None: 
        	e[2] = computed_list[curr_ind]
        	curr_ind += 1
        report_format = report_format + ' ' + e[0]
        report_value_list.append(e[2])
    return report_format[1:], report_value_list

def current_platform():
	from sys import platform
	return platform

def get_exp_dir(global_args):
	# if global_args.restore:
	if False:
		exp_dir = global_args.global_exp_dir + '/' + global_args.restore_dir
	else:
		random_name = str(uuid.uuid4().hex)
		exp_dir = global_args.global_exp_dir + '/' + random_name
		if len(global_args.exp_dir_postfix)>0: exp_dir = exp_dir + '_' + global_args.exp_dir_postfix 
	exp_dir = exp_dir+ '/'
	print('\n\nEXPERIMENT RESULT DIRECTORY: '+ exp_dir + '\n\n')

	filestring = 'Directory: '+exp_dir+ '\n\n Specs: \n\n'+ str(global_args)+'\n\n'
	if not os.path.exists(exp_dir): os.makedirs(exp_dir)
	with open(exp_dir+"Specs.txt", "w") as text_file:
	    text_file.write(filestring)
	return exp_dir

def list_hyperparameters(exp_folder):
    spec_file_path = exp_folder+'Specs.txt'
    target_file_path = exp_folder+'Listed_Specs.txt'
    with open(spec_file_path, "r") as text_file: data_lines = text_file.readlines()
    all_data_str = ''.join(data_lines)
    all_data_str = all_data_str.split('Namespace', 1)[-1]
    all_data_str = all_data_str.rstrip("\n")
    all_data_str = all_data_str[1:-1]
    
    split_list = all_data_str.split(',')
    full_list = []
    curr = []
    for e in split_list:
        if '=' in e:
            full_list.append(''.join(curr))
            curr = []
        curr.append(e)
    full_list = full_list[1:]
    pro_full_list = []
    for e in full_list: 
        pro_full_list.append(e.strip().replace("'", '"')) 
    pro_full_list.sort()
    with open(target_file_path, "w") as text_file: text_file.write('\n'.join(pro_full_list))

def debugger():
	import sys, ipdb, traceback
	def info(type, value, tb):
	   traceback.print_exception(type, value, tb)
	   print
	   ipdb.pm()
	sys.excepthook = info

def load_data_compressed(path):
    with open(path, 'rb') as f:
        return pickle.loads(zlib.decompress(f.read()))

def read_cropped_celebA(filename, size=64):
	rgb = scipy.misc.imread(filename)
	if size==64:
		crop_size = 108
		x_start = (rgb.shape[0]-crop_size)/2
		y_start = (rgb.shape[1]-crop_size)/2
		rgb_cropped = rgb[x_start:x_start+crop_size,y_start:y_start+crop_size,:]
		rgb_scaled = scipy.imresize(rgb_cropped, (size, size), interp='bilinear')
	return rgb_scaled

def split_tensor_np(tensor, dimension, list_of_sizes):
	split_list = []
	curr_ind = 0
	for s in list_of_sizes:		
		split_list.append(np.take(tensor, range(curr_ind, curr_ind+s), axis=dimension))
		curr_ind += s
	return split_list

def add_object_to_collection(x, name):
    if type(x) is list:
        for i in range(len(x)):
            tf.add_to_collection('list/'+name+'_'+str(i), x[i])
    else:
        tf.add_to_collection(name, x)


def slice_parameters(parameters, start, size):
    sliced_param = tf.slice(parameters, [0, start], [-1, size])
    new_start = start+size
    return sliced_param, new_start


def selu(x, lambda_var=1.0507009873554804934193349852946, alpha_var=1.6732632423543772848170429916717):
	positive_x = 0.5*(x+tf.abs(x))
	negative_x = 0.5*(x-tf.abs(x))
	negative = alpha_var*(tf.exp(negative_x)-1)
	return lambda_var*(positive_x+negative)

def lrelu(x, leak=0.2):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)

def parametric_relu(x, positive, negative):
         f1 = 0.5 * (positive + negative)
         f2 = 0.5 * (positive - negative)
         return f1 * x + f2 * tf.abs(x)

def polinomial_nonlin(x, coefficients):
	y_k = 0
	for order in range(coefficients.get_shape().as_list()[-1]):
		coefficient_batch_vector = coefficients[:,order][:, np.newaxis]
		pdb.set_trace()
		y_k = y_k+coefficient_batch_vector*x**order
	return y_k

def sigmoid_J(x):
	return (1-tf.sigmoid(x))*tf.sigmoid(x)

def tanh_J(x):
	return 1-tf.nn.tanh(x)**2

def relu_J(x):
	return (tf.abs(x)+x)/(2*x)

def parametric_relu_J(x, positive, negative):
	b_positive_x = (tf.abs(x)+x)/(2*x)
	return b_positive_x*positive+(1-b_positive_x)*negative

def polinomial_nonlin_J(x, coefficients):
	new_coefficients = coefficients[:, 1:]
	new_coefficients_dim = new_coefficients.get_shape().as_list()[-1]
	pdb.set_trace()
	new_coefficients = new_coefficients*tf.linspace(1.0, new_coefficients_dim, new_coefficients_dim)[np.newaxis, :]
	return polinomial_nonlin(x, coefficients)

class batch_norm(object):
	def __init__(self, epsilon=1e-5, decay = 0.9, name="batch_norm"):
		self.epsilon  = epsilon
		self.decay = decay
		self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
											data_format='NHWC', 
											decay=self.decay, 
											updates_collections=None,
											epsilon=self.epsilon,
											scale=True,
											is_training=train)

def conv_layer_norm_layer(input_layer, channel_index=3):
    input_layer_offset = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = input_layer.get_shape().as_list()[channel_index], use_bias = False, activation = None)[0]
    input_layer_scale = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = input_layer.get_shape().as_list()[channel_index], use_bias = False, activation = None)[0]
    return layer_norm(input_layer, [-1,-2,-3], channel_index, input_layer_offset, input_layer_scale)

def layer_norm(x, norm_axes, channel_index, channel_offset, channel_scale):
    # norm_axes = [-1,-2,-3]
    # norm_axes = [-1]
    mean, var = tf.nn.moments(x, norm_axes, keep_dims=True)
    frame = [1, 1, 1]
    frame[channel_index-1] = -1
    offset = tf.reshape(channel_offset, frame)
    scale = tf.reshape(channel_scale, frame)
    return tf.nn.batch_normalization(x, mean, var, offset, scale+1, 1e-5)

def get_object_from_collection(object_type, name):
    if object_type is list:
        out = []
        while True:
            try: out.append(tf.get_collection('list/'+name+'_'+str(len(out)))[0])
            except: break;
        return out
    else:
        return tf.get_collection(name)[0]

def split_tensor(tensor, dimension, list_of_sizes):
	split_list = []
	curr_ind = 0
	for s in list_of_sizes:
		split_list.append(tensor.narrow(dimension, curr_ind, s).contiguous())
		curr_ind += s
	return split_list

def split_tensor_tf(tensor, dimension, list_of_sizes):
	if tensor is None: return []
	if dimension == -1: dimension = len(tensor.get_shape().as_list())-1
	return tf.split(tensor, list_of_sizes, axis=dimension)

def list_sum(list_input):
	if len(list_input) == 0: return 0
	summed = None
	for e in list_input:
		if summed is None: summed = e
		else: summed = summed+e
	return summed

def pigeonhole_score(random_samples_from_model, subset=500, neigh=0.1):
	n_dim = np.prod(random_samples_from_model.shape[2:])
	max_euclidean_distance = np.sqrt(n_dim)
	random_samples_from_model_flat = random_samples_from_model.reshape(random_samples_from_model.shape[0]*random_samples_from_model.shape[1],-1)
	rates_of_matches = []
	for i in range(int(random_samples_from_model_flat.shape[0]/subset)):
		condensed_valid_distances = scipy.spatial.distance.pdist(random_samples_from_model_flat[i*subset:(i+1)*subset,:], metric='euclidean')
		rates_of_matches.append(np.mean((condensed_valid_distances<neigh*max_euclidean_distance)))
	return np.mean(rates_of_matches), np.std(rates_of_matches)

def list_product(list_input):
	if len(list_input) == 0: return 0
	else: return int(np.asarray(list_input).prod())

def list_merge(list_of_lists_input):
	out = []
	for e in list_of_lists_input:  out = out+e
	return out

def list_remove_none(list_input):
	return [x for x in list_input if x is not None]

def generate_empty_lists(num_lists, list_length):
	return [[None]*list_length for i in range(num_lists)]

def interleave_data(list_of_data):
	data_size = list_of_data[0].shape
	alldata = np.zeros((data_size[0]*len(list_of_data), *data_size[1:]))
	for i in range(len(list_of_data)):
		alldata[i::len(list_of_data)] = list_of_data[i]
	return  alldata

def extract_vis_data(reshape_func, obs, obs_param_out, batch_size):
    batch_reshaped = reshape_func(obs.view(-1, *obs.size()[2:]))
    batch_reshaped = [e.contiguous().view(obs.size(0), obs.size(1), *e.size()[1:]) for e in batch_reshaped]
    vis_data = [(batch_reshaped[i][:batch_size, ...].data.numpy(), 
                 obs_param_out[i][:batch_size, ...].data.numpy()) for i in range(len(obs_param_out))]
    return vis_data

def visualize_images(visualized_list, batch_size = 20, time_size = 30, save_dir = './', postfix = ''):
	visualized_list = visualized_list[..., :3]
	if not os.path.exists(save_dir): os.makedirs(save_dir)	
	batch_size = min(visualized_list.shape[0], batch_size)
	time_size = min(visualized_list.shape[1], time_size)
	block_size = [batch_size, time_size]
	padding = [5, 5]
	image_size = visualized_list.shape[-3:]
	canvas = np.ones([image_size[0]*block_size[0]+ padding[0]*(block_size[0]+1), 
					  image_size[1]*block_size[1]+ padding[1]*(block_size[1]+1), image_size[2]])

	for i in range(block_size[0]):
		start_coor = padding[0] + i*(image_size[0]+padding[0])
		for t in range(block_size[1]):
			y_start = (t+1)*padding[1]+t*image_size[1]
			canvas[start_coor:start_coor+image_size[0], y_start:y_start+image_size[1], :] =  visualized_list[i][t]
	if canvas.shape[2] == 1: canvas = np.repeat(canvas, 3, axis=2)
	scipy.misc.toimage(canvas).save(save_dir+'imageMatrix_'+postfix+'.png')

def visualize_images2(visualized_list, block_size, max_examples=20, save_dir = './', postfix = '', postfix2 = None):
	visualized_list = visualized_list[..., :3]
	assert(visualized_list.shape[0] == np.prod(block_size))
	padding = [1, 1]
	image_size = visualized_list.shape[-3:]
	canvas = np.ones([image_size[0]*min(block_size[0], max_examples)+ padding[0]*(min(block_size[0], max_examples)+1), 
					  image_size[1]*block_size[1]+ padding[1]*(block_size[1]+1), image_size[2]])

	visualized_list = visualized_list.reshape(*block_size, *visualized_list.shape[-3:])
	for i in range(min(block_size[0], max_examples)):
		start_coor = padding[0] + i*(image_size[0]+padding[0])
		for t in range(block_size[1]):
			y_start = (t+1)*padding[1]+t*image_size[1]
			canvas[start_coor:start_coor+image_size[0], y_start:y_start+image_size[1], :] =  visualized_list[i][t]
	if canvas.shape[2] == 1: canvas = np.repeat(canvas, 3, axis=2)

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	scipy.misc.toimage(canvas).save(save_dir+'imageMatrix_'+postfix+'.png')
	# if postfix2 is None: scipy.misc.toimage(canvas).save(save_dir+'/../imageMatrix.png')
	# else: scipy.misc.toimage(canvas).save(save_dir+'/../imageMatrix_'+postfix2+'.png')
	if postfix2 is not None: scipy.misc.toimage(canvas).save(save_dir+'/../imageMatrix_'+postfix2+'.png')

	# canvas_int = (canvas*255).astype('uint8')
	# imsave(save_dir+'imageMatrix_'+postfix+'_2.png', canvas_int)
	# from PIL import Image
	# result = Image.fromarray(canvas_int)
	# result.save(save_dir+'imageMatrix_'+postfix+'_3.png', format='JPEG', subsampling=0, quality=100)

def draw_quivers(data, path):
	max_value =  np.abs(data).max()
	# max_value =  0.5
	soa = np.concatenate((np.zeros(data.T.shape), data.T),axis = 1)
	X, Y, U, V = zip(*soa)
	plt.figure()
	ax = plt.gca()
	colors = np.asarray([[1, 0, 0, 1], [0, 1, 0., 1], [0, 0, 1., 1]])
	ax.quiver(X, Y, U, V, color = colors, angles='xy', scale_units='xy', scale=1)
	ax.set_xlim([-max_value, max_value])
	ax.set_ylim([-max_value, max_value])
	plt.savefig(path)
	plt.close('all')

def draw_bar_plot(vector, y_min_max = None, thres = None, save_dir = './', postfix = ''):
	fig = plt.figure()
	plt.cla()
	plt.bar(np.arange(vector.shape[0]), vector, align='center')

	if thres is not None:
		plt.plot([0., vector.shape[0]], [thres[0], thres[0]], "r--")
		plt.plot([0., vector.shape[0]], [thres[1], thres[1]], "g--")
	plt.xlim(0, vector.shape[0])
	if y_min_max is not None:
		plt.ylim(*y_min_max)

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	plt.savefig(save_dir+'barplot_'+postfix+'.png')
	plt.close('all')

def plot_ffs(xx, yy, f, save_dir = './', postfix = ''):
	# import matplotlib.mlab as mlab
	# delta = 0.025
	# x = np.arange(-3.0, 3.0, delta)
	# y = np.arange(-2.0, 2.0, delta)
	# X, Y = np.meshgrid(x, y)
	# Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
	# Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
	# # difference of Gaussians
	# Z = 10.0 * (Z2 - Z1)


	# # Create a simple contour plot with labels using default colors.  The
	# # inline argument to clabel will control whether the labels are draw
	# # over the line segments of the contour, removing the lines beneath
	# # the label
	# plt.figure()
	# CS = plt.contour(X.flatten(), Y.flatten(), Z.flatten())
	# plt.clabel(CS, inline=1, fontsize=10)
	# plt.title('Simplest default with labels')
	# plt.savefig('barplot_'+postfix+'_2.png')
	# pdb.set_trace()

	fig = plt.figure(figsize=(15, 15), dpi=300)
	plt.cla()
	plt.imshow(np.flipud(f), cmap='rainbow', interpolation='none')
	# # plt.xlim(0, vector.shape[0])
	# # if y_min_max is not None:
	# # 	plt.ylim(*y_min_max)
	if not os.path.exists(save_dir+'barplot/'): os.makedirs(save_dir+'barplot/')
	plt.savefig(save_dir+'barplot/barplot_'+postfix+'.png')
	plt.close('all')

	fig = plt.figure(figsize=(15, 15), dpi=300)
	plt.cla()
	heatmap = plt.pcolor(xx, yy, f, cmap='RdBu', vmin=np.min(f), vmax=np.max(f))
	plt.colorbar(heatmap)
	if not os.path.exists(save_dir+'barplot_3/'): os.makedirs(save_dir+'barplot_3/')
	plt.savefig(save_dir+'barplot_3/barplot_3_'+postfix+'.png')
	plt.close('all')


	fig = plt.figure(figsize=(15, 15), dpi=300)
	plt.cla()
	# plt.imshow(f, cmap='rainbow', interpolation='none')
	ax = fig.gca()
	cfset = ax.contourf(xx, yy, f, cmap='Blues')
	# # plt.xlim(0, vector.shape[0])
	# # if y_min_max is not None:
	# # 	plt.ylim(*y_min_max)
	if not os.path.exists(save_dir+'barplot_2/'): os.makedirs(save_dir+'barplot_2/')
	plt.savefig(save_dir+'barplot_2/barplot_2_'+postfix+'.png')
	plt.close('all')

def visualize_flat(visualized_list, batch_size = 10, save_dir = './', postfix = ''):
	canvas = -1*np.ones((int(len(visualized_list)/3)*4, max([e.shape[2] for e in visualized_list[:int(len(visualized_list)/3)]])))
	num_items = int(len(visualized_list)/3)
	cont_item = num_items-2
	time_step = -1
	for b in range(min(batch_size, visualized_list[0].shape[0])):
		canvas.fill(-1)
		for i in range(num_items):
			mat = np.concatenate([visualized_list[i][..., np.newaxis], 
								  visualized_list[i+int(len(visualized_list)/3)][..., np.newaxis],
								  visualized_list[i+2*int(len(visualized_list)/3)][..., np.newaxis]], axis = -1)
			data = mat[b][time_step]
			canvas[i*4:i*4+3, 0:data.shape[0]] = data.T
		plt.matshow(canvas)
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		plt.savefig(save_dir+'flatMatrix_'+str(b)+'_'+str(time_step)+'_'+postfix+'.png')
		plt.close('all')

def visualize_flat2(visualized_list, batch_size = 10, save_dir = './', postfix = ''):
	time_step = -1
	width = 0.35
	num_dimensions_to_visualize = min(50, visualized_list[1][0][time_step].shape[0])
	ind = np.arange(num_dimensions_to_visualize)  # the x locations for the groups
	for i in range(batch_size):
		rects1 = plt.bar(ind, visualized_list[0][i][time_step][:num_dimensions_to_visualize], width, color='b')
		rects2 = plt.bar(ind + width, visualized_list[1][i][time_step][:num_dimensions_to_visualize], width, color='g')
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		plt.savefig(save_dir+'flatMatrix_'+str(i)+'_'+str(time_step)+'_'+postfix+'.png')
		plt.close('all')

def dataset_plotter_old(data_list, save_dir = './', postfix = '', postfix2 = None, show_also = False):
	colors = ['r', 'g', 'b', 'k']
	alphas = [1, 0.3, 0.3, 1.]
	if data_list[0].shape[1]==3:
		fig = plt.figure()
		plt.cla()
		ax = p3.Axes3D(fig)
		ax.view_init(7, -80)
		for i, data in enumerate(data_list):
			ax.plot3D(data[:, 0], data[:, 1], data[:, 2], 'o', color = colors[i])

	if data_list[0].shape[1]==2:
		fig = plt.figure(figsize=(8, 8), dpi=150)
		plt.cla()
		ax = fig.gca()
		# xmin, xmax = -3.5, 3.5
		# ymin, ymax = -3.5, 3.5

		xmin, xmax = -1.5, 1.5
		ymin, ymax = -1.5, 1.5

		# if len(data_list)==3: pdb.set_trace()
		for i, data in enumerate(data_list):			
			plt.scatter(data[:, 0], data[:, 1], color = colors[i], s=0.5, alpha=alphas[i]) #, edgecolors='none'
			# plt.scatter(data_list[i][:, 0], data_list[i][:, 1], color = colors[i], s=0.5, alpha=alphas[i]) #, edgecolors='none'
		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		plt.axes().set_aspect('equal')

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	plt.savefig(save_dir+'datasets_'+postfix+'.png')
	if postfix2 is None: plt.savefig(save_dir+'/../datasets.png')
	else: plt.savefig(save_dir+'/../datasets_'+postfix2+'.png')
	# if postfix2 == 'data_only_3': pdb.set_trace()

	if show_also: plt.show()
	else: plt.close('all')

def dataset_plotter(data_list, ranges=None, tie=False, point_thickness=0.5, colors=None, save_dir = './', postfix = '', postfix2 = None, show_also = False):
	if colors is None: colors = ['r', 'g', 'b', 'k', 'c', 'm', 'gold', 'teal', 'springgreen', 'lightcoral', 'darkgray']
	# alphas = [1, 0.3, 0.3, 1.]
	# alphas = [1, 0.6, 0.4, 1.]
	alphas = [0.3, 1, 1, 1.]

	# if data_list[0].shape[1]==3:
	# 	fig = plt.figure()
	# 	plt.cla()
	# 	ax = p3.Axes3D(fig)
	# 	ax.view_init(11, 0)
	# 	for i, data in enumerate(data_list):
	# 		ax.plot3D(data[:, 0], data[:, 1], data[:, 2], 'o', color = colors[i])
	n_lines = 50

	if data_list[0].shape[1]==3:
		fig = plt.figure(figsize=(8, 8))
		plt.cla()
		ax = fig.add_subplot(111, projection='3d')
		# ax.set_title('z-axis left side')
		ax = fig.add_axes(MyAxes3D(ax, 'l'))
		# example_surface(ax) # draw an example surface
		# cm = plt.get_cmap("RdYlGn")
		
		# fig = plt.figure()
		# plt.cla()
		# ax = p3.Axes3D(fig)
		# ax.view_init(11, 0)
		for i, data in enumerate(data_list):
			X, Y, Z = data[:, 0], data[:, 1], data[:, 2]
			# ax.scatter(X,Y,Z, c=cm.coolwarm(Z), linewidth=0)
			ax.scatter(X,Y,Z, color = colors[i], linewidth=0)
			# ax.plot3D(X,Y,Z, 'o', facecolors=cm.jet(Z)) #color = colors[i])


			# squareshape = [int(np.sqrt(X.shape[0])), int(np.sqrt(X.shape[0]))]
			# X = X.reshape(squareshape)
			# Y = Y.reshape(squareshape)
			# Z = Z.reshape(squareshape)
			# surf = ax.plot_surface(data[:, 0], data[:, 1], data[:, 2], rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			# surf = ax.plot_surface(data[:, 0], data[:, 1], data[:, 2], 'o', color = colors[i], antialiased=False)#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			# surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm, linewidth=1, antialiased=False)#,color = cm.coolwarm)

			# ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.3)
			# cset = ax.contour(X, Y, Z, zdir='z', offset=-15, cmap=cm.coolwarm)
			# cset = ax.contour(X, Y, Z, zdir='x', offset=-15, cmap=cm.coolwarm)
			# cset = ax.contour(X, Y, Z, zdir='y', offset=15, cmap=cm.coolwarm)


		# 	surf = ax.plot_surface(X, Y, Z, rstride=3, cstride=3, alpha=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
		# fig.colorbar(surf, shrink=0.5, aspect=5)
			
		# ax.set_zlim(-1.01, 1.01)
		# ax.zaxis.set_major_locator(LinearLocator(10))
		# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

		
		# Add a color bar which maps values to colors.
		# fig.colorbar(surf, shrink=0.5, aspect=5)

		if ranges is None: ranges = (-5, 5)
		ax.set_xlim(*ranges)
		ax.set_ylim(*ranges)
		ax.set_zlim(*ranges)

	if data_list[0].shape[1]==2:
		fig = plt.figure(figsize=(8, 8), dpi=150)
		plt.cla()
		ax = fig.gca()
		xmin, xmax = -3., 3.
		ymin, ymax = -3., 3.
		# if len(data_list)==3: pdb.set_trace()
		for i, data in enumerate(data_list):
			if len(data_list)<=4: 			
				plt.scatter(data[:, 0], data[:, 1], color = colors[i], s=point_thickness, alpha=alphas[i]) #, edgecolors='none'
			else:
				plt.scatter(data[:, 0], data[:, 1], color = colors[i], s=3) #, edgecolors='none'
			# plt.scatter(data_list[i][:, 0], data_list[i][:, 1], color = colors[i], s=0.5, alpha=alphas[i]) #, edgecolors='none'
		if tie and len(data_list)==3:
			norms = np.sum((data_list[1]-data_list[2])**2, axis=1)
			norm_order = np.argsort(norms)[::-1]
			# norm_order = np.argsort(norms)

			x_all, y_all = [], []
			for j in range(min(data_list[1].shape[0], n_lines)):
				# print(j, norm_order[j])
				x1, y1 = [data_list[1][norm_order[j]][0], data_list[2][norm_order[j]][0]], [data_list[1][norm_order[j]][1], data_list[2][norm_order[j]][1]]
				# np.concatenate([data_list[0][norm_order,0,np.newaxis], data_list[1][norm_order,0,np.newaxis]],axis=1)[:2,:]
				# np.concatenate([data_list[0][:,0,np.newaxis], data_list[1][:,0,np.newaxis]],axis=1)
				# pdb.set_trace()
				plt.plot(x1, y1, 'k')

		ax.set_xlim([xmin, xmax])
		ax.set_ylim([ymin, ymax])
		plt.axes().set_aspect('equal')

	ax = plt.gca()
	try:
		ax.set_axis_bgcolor((1., 1., 1.))
	except:
		ax.set_facecolor((1., 1., 1.))

	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['left'].set_color('black')

	if not os.path.exists(save_dir): os.makedirs(save_dir)
	plt.savefig(save_dir+'datasets_'+postfix+'.png', bbox_inches='tight')
	if postfix2 is None: plt.savefig(save_dir+'/../datasets.png', bbox_inches='tight')
	else: plt.savefig(save_dir+'/../datasets_'+postfix2+'.png', bbox_inches='tight')

	if show_also: plt.show()
	else: plt.close('all')


def plot2D_dist(data, save_dir = './', postfix = ''):
	x = data[:, 0]
	y = data[:, 1]
	# xmin, xmax = x.min()-0.1, x.max()+0.1
	# ymin, ymax = y.min()-0.1, y.max()+0.1
	
	# xmin, xmax = -3.5, 3.5
	# ymin, ymax = -3.5, 3.5

	xmin, xmax = -1.5, 1.5
	ymin, ymax = -1.5, 1.5

	# Peform the kernel density estimate
	xx, yy = np.mgrid[xmin:xmax:125j, ymin:ymax:125j]
	positions = np.vstack([xx.ravel(), yy.ravel()])
	values = np.vstack([x, y])
	kernel = st.gaussian_kde(values)
	f = np.reshape(kernel(positions).T, xx.shape)

	fig = plt.figure(figsize=(8, 8), dpi=150)
	ax = fig.gca()
	# Contourf plot

	cfset = ax.contourf(xx, yy, f, cmap='Blues')
	## Or kernel density estimate plot instead of the contourf plot
	#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
	# Contour plot
	print(len(xx))
	print(xx.shape)
	print(len(yy))
	print(yy.shape)
	print(len(f))
	print(f.shape)
	cset = ax.contour(xx, yy, f, colors='k')
	# Label plot
	ax.set_xlabel('Y1')
	ax.set_ylabel('Y0')
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])

	plt.axes().set_aspect('equal')
	if not os.path.exists(save_dir+'/plot2D_dist/'): os.makedirs(save_dir+'/plot2D_dist/')
	plt.savefig(save_dir+'plot2D_dist/distribution_'+postfix+'.png')
	plt.savefig(save_dir+'distribution.png')
	ax.clabel(cset, inline=1, fontsize=10)

	if not os.path.exists(save_dir+'/plot2D_dist_b_labeled/'): os.makedirs(save_dir+'/plot2D_dist_b_labeled/')
	plt.savefig(save_dir+'plot2D_dist_b_labeled/distribution_b_labeled_'+postfix+'.png')
	plt.savefig(save_dir+'distribution_b_labeled.png')
	plt.close('all')


def plot2D_dist2(Y):
	Y = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
	ax = sns.kdeplot(Y, shade = True, cmap = "PuBu")
	ax.patch.set_facecolor('white')
	ax.collections[0].set_alpha(0)
	ax.set_xlabel('$Y_1$', fontsize = 15)
	ax.set_ylabel('$Y_0$', fontsize = 15)
	plt.xlim(-3, 3)
	plt.ylim(-3, 3)
	plt.plot([-3, 3], [-3, 3], color = "black", linewidth = 1)
	plt.savefig('./ff_dist2.png')
	# plt.show()
	pdb.set_trace()

def visualize_datasets(sess, feed_dict, dataset, obs_sample_out_tf, latent_sample_out_tf, transported_sample_out_tf=None, input_sample_out_tf=None, save_dir = './', postfix = ''):
	n_sampled = 0
	all_obs_sample_out = None
	all_latent_sample_out = None
	all_transp_sample_out = None
	all_input_sample_out = None

	while n_sampled < dataset.shape[0]:
		try: input_sample_out, transported_sample_out, obs_sample_out, latent_sample_out = sess.run([input_sample_out_tf['flat'], transported_sample_out_tf['flat'], obs_sample_out_tf['flat'], latent_sample_out_tf], feed_dict = feed_dict)
		except: obs_sample_out, latent_sample_out = sess.run([obs_sample_out_tf['flat'], latent_sample_out_tf], feed_dict = feed_dict)

		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		if all_latent_sample_out is None: all_latent_sample_out = latent_sample_out.reshape(-1, latent_sample_out.shape[-1])
		else: all_latent_sample_out = np.concatenate([all_latent_sample_out, latent_sample_out.reshape(-1, latent_sample_out.shape[-1])], axis=0)
		try: 
			if all_transp_sample_out is None: all_transp_sample_out = transported_sample_out.reshape(-1, transported_sample_out.shape[-1])
			# else: all_transp_sample_out = np.concatenate([all_transp_sample_out, transported_sample_out.reshape(-1, transported_sample_out.shape[-1])], axis=0)
			if all_input_sample_out is None: all_input_sample_out = input_sample_out.reshape(-1, input_sample_out.shape[-1])
			# else: all_input_sample_out = np.concatenate([all_input_sample_out, input_sample_out.reshape(-1, input_sample_out.shape[-1])], axis=0)

		except: pass

		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]

	all_obs_sample_out = all_obs_sample_out[:dataset.shape[0], ...]
	all_latent_sample_out = all_latent_sample_out[:dataset.shape[0], ...]

	# try: 
	# 	all_transp_sample_out = all_transp_sample_out[:dataset.shape[0], ...]
	# 	all_input_sample_out = all_input_sample_out[:dataset.shape[0], ...]
	# except: pass

	print('Mean: ', all_obs_sample_out[:,0].mean(), all_obs_sample_out[:,1].mean())
	print('Variance: ', all_obs_sample_out[:,0].std(), all_obs_sample_out[:,1].std())

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only/', postfix = postfix+'_data_only', postfix2 = 'data_only')
	dataset_plotter([dataset, all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_real/', postfix = postfix+'_data_real', postfix2 = 'data_real')
	try: 
		dataset_plotter([dataset, all_input_sample_out, all_transp_sample_out], save_dir = save_dir+'/dataset_plotter_data_transport_lined/', tie=True, postfix = postfix+'_data_transport_lined', postfix2 = 'data_transport_lined')
		dataset_plotter([dataset, all_transp_sample_out], save_dir = save_dir+'/dataset_plotter_data_transport/', postfix = postfix+'_data_transport', postfix2 = 'data_transport')
	except: pass

	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def visualizeTransitions(sess, input_dict, generative_dict, save_dir = '.', postfix = ''):
	interpolated_sample_np, interpolated_further_sample_np, interpolated_sample_begin_np, interpolated_sample_end_np = \
		sess.run([generative_dict['interpolated_sample']['image'], generative_dict['interpolated_further_sample']['image'], 
				  generative_dict['interpolated_sample_begin']['image'], generative_dict['interpolated_sample_end']['image']], feed_dict = input_dict)
	
	interpolated_sample_linear_np = sess.run(generative_dict['interpolated_sample_linear']['image'], feed_dict = input_dict)
	# samples_params_np = np.array([interpolated_sample_begin_np, interpolated_sample_np, interpolated_further_sample_np, interpolated_sample_end_np])
	samples_params_np = np.array([interpolated_sample_begin_np, *interpolated_sample_linear_np, interpolated_sample_end_np])
	vis_data = np.concatenate(samples_params_np, axis=1)
	np.clip(vis_data, 0, 1, out=vis_data)
	visualize_images(np.concatenate(samples_params_np, axis=1), save_dir = save_dir, postfix = postfix+'_'+'image')
	visualize_images(vis_data, save_dir = save_dir, postfix = postfix+'_'+'image_2')

def visualize_datasets2(sess, feed_dict_func, data_loader, dataset, obs_sample_out_tf, save_dir = './', postfix = ''):
	data_loader.train()
	all_obs_sample_out, batch_all = None, None
	n_sampled, num_batches, i = 0, 10, 0
	for batch_idx, curr_batch_size, batch in data_loader:
		if i == num_batches: break
		if batch_all is None: batch_all = copy.deepcopy(batch) 
		else:
			try: batch_all['observed']['data']['flat'] = np.concatenate([batch_all['observed']['data']['flat'], batch['observed']['data']['flat']], axis=0)
			except: batch_all['observed']['data']['image'] = np.concatenate([batch_all['observed']['data']['image'], batch['observed']['data']['image']], axis=0)
		i += 1
	feed_dict = feed_dict_func(batch_all)

	while n_sampled < dataset.shape[0]:
		obs_sample_out = sess.run(obs_sample_out_tf['flat'], feed_dict = feed_dict)
		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]
	all_obs_sample_out = all_obs_sample_out[:dataset.shape[0], ...]

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only_2/', postfix = postfix+'_data_only_2', postfix2 = 'data_only_2')
	dataset_plotter([all_obs_sample_out, dataset], save_dir = save_dir+'/dataset_plotter_data_real_2/', postfix = postfix+'_data_real_2', postfix2 = 'data_real_2')
	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def visualize_datasets3(sess, feed_dict_func, data_loader, dataset, obs_sample_out_tf, save_dir = './', postfix = ''):
	n_sampled = 0
	all_obs_sample_out = None
	data_loader.train()
	for batch_idx, curr_batch_size, batch in data_loader:
		obs_sample_out = sess.run(obs_sample_out_tf['flat'], feed_dict = feed_dict_func(batch))
		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]
	all_obs_sample_out = all_obs_sample_out[:n_sampled, ...]

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only_3/', postfix = postfix+'_data_only_3', postfix2 = 'data_only_3')
	dataset_plotter([all_obs_sample_out, dataset], save_dir = save_dir+'/dataset_plotter_data_real_3/', postfix = postfix+'_data_real_3', postfix2 = 'data_real_3')
	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def visualize_datasets4(sess, feed_dict_func, data_loader, obs_sample_begin_tf, obs_sample_end_tf, obs_sample_out_tf, save_dir = './', postfix = ''):
	data_loader.train()
	all_obs_sample_begin, all_obs_sample_end, all_obs_sample_out, batch_all = None, None, None, None
	n_sampled, num_batches, i = 0, 1, 0
	for batch_idx, curr_batch_size, batch in data_loader:
		if i == num_batches: break
		if batch_all is None: batch_all = copy.deepcopy(batch) 
		else:
			try: batch_all['observed']['data']['flat'] = np.concatenate([batch_all['observed']['data']['flat'], batch['observed']['data']['flat']], axis=0)
			except: batch_all['observed']['data']['image'] = np.concatenate([batch_all['observed']['data']['image'], batch['observed']['data']['image']], axis=0)
		i += 1
	feed_dict = feed_dict_func(batch_all)

	while n_sampled < 20000:
		obs_sample_begin, obs_sample_end, obs_sample_out = sess.run([obs_sample_begin_tf['flat'], obs_sample_end_tf['flat'], obs_sample_out_tf['flat']], feed_dict = feed_dict)
		if all_obs_sample_begin is None: all_obs_sample_begin = obs_sample_begin.reshape(-1, obs_sample_begin.shape[-1])
		else: all_obs_sample_begin = np.concatenate([all_obs_sample_begin, obs_sample_begin.reshape(-1, obs_sample_begin.shape[-1])], axis=0)
		if all_obs_sample_end is None: all_obs_sample_end = obs_sample_end.reshape(-1, obs_sample_end.shape[-1])
		else: all_obs_sample_end = np.concatenate([all_obs_sample_end, obs_sample_end.reshape(-1, obs_sample_end.shape[-1])], axis=0)
		if all_obs_sample_out is None: all_obs_sample_out = obs_sample_out.reshape(-1, obs_sample_out.shape[-1])
		else: all_obs_sample_out = np.concatenate([all_obs_sample_out, obs_sample_out.reshape(-1, obs_sample_out.shape[-1])], axis=0)
		n_sampled += obs_sample_out.reshape(-1, obs_sample_out.shape[-1]).shape[0]

	dataset_plotter([all_obs_sample_out], save_dir = save_dir+'/dataset_plotter_data_only_4/', postfix = postfix+'_data_only_4', postfix2 = 'data_only_4')
	dataset_plotter([all_obs_sample_out, all_obs_sample_begin, all_obs_sample_end], save_dir = save_dir+'/dataset_plotter_data_real_4/', postfix = postfix+'_data_real_4', postfix2 = 'data_real_4')
	# plot2D_dist(all_obs_sample_out, save_dir = save_dir, postfix = postfix)

def safe_tf_sqrt(x, clip_value=1e-5):
	return tf.sqrt(tf.clip_by_value(x, clip_value, np.inf))

def variable_summaries(var, name):
	mean = tf.reduce_mean(var)
	tf.summary.scalar('mean/' + name, mean)
	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev/' + name, stddev)
	tf.summary.scalar('max/' + name, tf.reduce_max(var))
	tf.summary.scalar('min/' + name, tf.reduce_min(var))
	tf.summary.histogram(name, var)

def visualize_vectors(visualized_list, batch_size = 10, save_dir = './', postfix = ''):
	time_step = -1
	for i in range(int(len(visualized_list)/3.)):
		mat = np.concatenate([visualized_list[i][..., np.newaxis], 
				  visualized_list[i+int(len(visualized_list)/3)][..., np.newaxis],
				  visualized_list[i+2*int(len(visualized_list)/3)][..., np.newaxis]], axis = -1)
		for b in range(min(batch_size, visualized_list[0].shape[0])):
			data = mat[b][time_step]
			if not os.path.exists(save_dir): os.makedirs(save_dir)
			draw_quivers(data, save_dir+'flatRelPos_obs_'+str(i)+'_example_'+str(b)+'_t_'+str(time_step)+'_'+postfix+'.png')

def copy_dir2(dir1, dir2):	
	if os.path.exists(dir1) and not os.path.exists(dir2): os.makedirs(dir2)
	file_list = glob.glob(dir1+'/*')
	for e in file_list:
		pdb.set_trace()

		shutil.copyfile(e, dir2+e[len(dir1):])

def copy_dir(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def save_checkpoint(saver, sess, global_step, exp_dir):
	if not os.path.exists(exp_dir): os.makedirs(exp_dir)
	# checkpoint_file = os.path.join(exp_dir , 'model.ckpt')
	# saver.save(sess, exp_dir+'/model' global_step=global_step)
	# saver.export_meta_graph(os.path.join(exp_dir , 'model.meta'))
	saver.save(sess, exp_dir+'/model')
	saver.export_meta_graph(exp_dir+'/model.meta')


def load_checkpoint(saver, sess, exp_dir):	
	# saver = tf.train.import_meta_graph(os.path.join(exp_dir , 'model.meta'))
	saver.restore(sess, exp_dir+'model')

	# ckpt = tf.train.get_checkpoint_state(exp_dir+'/checkpoints')
	# if ckpt and ckpt.model_checkpoint_path:
	# 	saver.restore(sess, ckpt.model_checkpoint_path)
	# 	print('Loaded checkpoint from file: ' + ckpt.model_checkpoint_path)
	# else: print('No Checkpoint..')


def visualize_categorical_data(vis_data, save_dir):
	for i in range(len(vis_data)):
		if i+1 == len(vis_data):
			real_action_map = vis_data[i][0][:, :-1].reshape(-1, 1, 10, 10)
			prob_action_map = vis_data[i][1][:, :-1].reshape(-1, 1, 10, 10)
			visualize_images([real_action_map, prob_action_map], save_dir+'/vis_cat_2/', [1, 10, 10])
		else:
			plt.cla()
			f, axes = plt.subplots(vis_data[i][0].shape[0], sharex=True, sharey=True, figsize=(8, 15))
			for j, axis in enumerate(axes):
				sample = vis_data[i][0][j]
				probs = vis_data[i][1][j]	
				markers,stems,base = axis.stem(sample, 'r', markerfmt='ro', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				markers,stems,base = axis.stem(probs, 'g', markerfmt='go', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				axis.axhline(0, color='blue', lw=2)

			f.subplots_adjust(hspace=0.3)
			plt.savefig(save_dir+'/vis_cat_2/VAEVis_'+str(i)+'.png')
	plt.close('all')

def visualize_categorical_data_series(vis_data, save_dir, time_index = 0, postfix = ''):
	if not os.path.exists(save_dir+'vis/'): os.makedirs(save_dir+'vis/')
	
	for i in range(len(vis_data)):
		category = vis_data[i]
		if i+1 == len(vis_data):
			real_action_map = category[0][:, time_index, :-1].reshape(-1, 1, 10, 10)
			prob_action_map = category[1][:, time_index, :-1].reshape(-1, 1, 10, 10)
			visualize_images([real_action_map, prob_action_map], save_dir+'vis/', [1, 10, 10], postfix = postfix)
		else:
			plt.cla()
			f, axes = plt.subplots(category[0].shape[0], sharex=True, sharey=True, figsize=(8, 15))
			for j, axis in enumerate(axes):
				sample = category[0][j]
				probs = category[1][j]	
				markers,stems,base = axis.stem(sample[time_index], 'r', markerfmt='ro', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				markers,stems,base = axis.stem(probs[time_index], 'g', markerfmt='go', bottom=0)
				for stem in stems: stem.set_linewidth(6)
				axis.axhline(0, color='blue', lw=2)

			f.subplots_adjust(hspace=0.3)
			plt.savefig(save_dir+'vis/'+'VAEVis_'+str(i)+'_'+postfix+'.png')
	plt.close('all')

class TemperatureObjectTF:
	def __init__(self, start_temp, max_steps):
		self.start_temp = start_temp
		self.max_steps = max_steps
		self.train()

	def train(self):
		self.mode = 'Train'

	def eval(self):
		self.mode = 'Test'
	
	def temp_step(self, t):
		if self.mode == 'Test': return 1.0
		return tf.minimum(1.0, self.start_temp+t/float(self.max_steps))

class TemperatureObject:
	def __init__(self, start_temp, max_steps):
		self.start_temp = start_temp
		self.max_steps = max_steps
		self.t = 0
		self.train()

	def train(self):
		self.mode = 'Train'

	def eval(self):
		self.mode = 'Test'
	
	def temp(self):
		if self.mode == 'Test': return 1.0
		return min(1.0, self.start_temp+float(self.t)/float(self.max_steps))

	def temp_step(self):
		curr_temp = self.temp()
		if self.mode == 'Train': self.t += 1
		return curr_temp

def serialize_model(model, path=None):
	pass

def deserialize_model(path, model=None):
	pass


def update_dict_from_file(dict_to_update, filename):
	try: 
		with open(filename) as f:
			content = f.readlines()
		for line in content:
			if line[0] != '#':  
				line_list = line.split(':')
				if len(line_list)==2 and line_list[0] in dict_to_update:
					try: dict_to_update[line_list[0]] = float(line_list[1])
					except: pass
	except: pass
	return dict_to_update

class PrintSnooper:
    def __init__(self, stdout):
        self.stdout = stdout
    def write(self, s):
        self.stdout.write('====print====\n')
        traceback.print_stack()
        self.stdout.write(s)
        self.stdout.write("\n")
    def flush(self):
        self.stdout.flush()

def tf_batch_and_input_dict(batch_template, additional_inputs_template=None):
    batch_tf = {}
    for d in ['context', 'observed']:
        if d not in batch_tf: batch_tf[d] = {}
        for a in ['properties', 'data']:
            if a not in batch_tf[d]: batch_tf[d][a] = {}
            for t in ['flat', 'image']:
                if a == 'properties': batch_tf[d][a][t] = batch_template[d][a][t]
                elif a == 'data': 
                    if batch_template[d][a][t] is None: batch_tf[d][a][t] = None
                    else: 
                    	batch_tf[d][a][t] = tf.placeholder(tf.float32, [None, *batch_template[d][a][t].shape[1:]]) 
                    	# batch_tf[d][a][t] = tf.placeholder(tf.float32, [*batch_template[d][a][t].shape]) 

    def input_dict_func(batch, additional_inputs=None):
        input_dict = {}
        for d in ['context', 'observed']:
            for t in batch_tf[d]['data']:
                if batch_tf[d]['data'][t] is not None:
                    input_dict[batch_tf[d]['data'][t]] = batch[d]['data'][t]
        if additional_inputs is not None: input_dict = {additional_inputs_template: additional_inputs, **input_dict}
        return input_dict
    return batch_tf, input_dict_func
 
def _beta_samples(full_size, alpha=0.5, beta=0.5):
	assert (alpha>0)
	assert (beta>0)
	return np.random.beta(alpha, beta, size=full_size).astype(np.float32)

def beta_samples(full_size, alpha=0.5, beta=0.5):
    return tf.py_func(_beta_samples, [full_size, alpha, beta], [tf.float32], name="beta_samples", stateful=False)[0]

def _triangular_ones(full_size, trilmode = 0):
  return np.tril(np.ones(full_size), trilmode).astype(np.float32)

def _block_triangular_ones(full_size, block_size, trilmode = 0):
  O = np.ones(block_size) 
  Z = np.zeros(block_size) 
  string_matrix = np.empty(full_size, dtype="<U3")
  string_matrix[:] = 'O'
  string_matrix = np.tril(string_matrix, trilmode)
  string_matrix[string_matrix == ''] = 'Z'
  stringForBlocks = ''
  for i in range(string_matrix.shape[0]):
    for j in range(string_matrix.shape[1]):
      stringForBlocks = stringForBlocks + string_matrix[i,j]
      if j!=string_matrix.shape[1]-1: stringForBlocks = stringForBlocks + ','
    if i!=string_matrix.shape[0]-1: stringForBlocks = stringForBlocks + ';'
  return np.bmat(stringForBlocks).astype(np.float32)

def _block_diagonal_ones(full_size, block_size):
  O = np.ones(block_size) 
  Z = np.zeros(block_size) 
  string_matrix = np.empty(full_size, dtype="<U3")
  string_matrix[:] = 'Z'
  np.fill_diagonal(string_matrix, 'O')
  stringForBlocks = ''
  for i in range(string_matrix.shape[0]):
    for j in range(string_matrix.shape[1]):
      stringForBlocks = stringForBlocks + str(string_matrix[i,j])
      if j!=string_matrix.shape[1]-1: stringForBlocks = stringForBlocks + ','
    if i!=string_matrix.shape[0]-1: stringForBlocks = stringForBlocks + ';'
  return np.bmat(stringForBlocks).astype(np.float32)

def triangular_ones(full_size, trilmode = 0):
    return tf.py_func(_triangular_ones, [full_size, trilmode], [tf.float32], name="triangular_ones", stateful=False)[0]

def block_triangular_ones(full_size, block_size, trilmode = 0):
    return tf.py_func(_block_triangular_ones, [full_size, block_size, trilmode], 
                      [tf.float32], name="block_triangular_ones", stateful=False)[0]

def block_diagonal_ones(full_size, block_size):
    return tf.py_func(_block_diagonal_ones, [full_size, block_size],
                      [tf.float32], name="block_diagonal_ones", stateful=False)[0]

def tf_remove_diagonal(tensor, first_dim_size=5, second_dim_size=5):
	lower_triangular_mask = triangular_ones([first_dim_size, second_dim_size], trilmode = -1)
	for i in range(len(tensor.get_shape())-2): lower_triangular_mask = lower_triangular_mask[..., np.newaxis]
	upper_triangular_mask = 1-triangular_ones([first_dim_size, second_dim_size], trilmode = 0)
	for i in range(len(tensor.get_shape())-2): upper_triangular_mask = upper_triangular_mask[..., np.newaxis]

	tensor_lt = (tensor*lower_triangular_mask)[1:,...]
	tensor_ut = (tensor*upper_triangular_mask)[:-1,...]
	tensor_without_diag = tensor_lt+tensor_ut
	return tensor_without_diag

# random_data = tf.random_normal((2, 2, 3, 3), 0, 1, dtype=tf.float32)
# total = tf_remove_diagonal(random_data, first_dim_size=random_data.get_shape().as_list()[0], second_dim_size=random_data.get_shape().as_list()[1])

# # init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# # sess.run(init)
# # out1, out2, out3, out4 = sess.run([random_data_lt, random_data_ut, total, random_data])
# out1, out2 = sess.run([total, random_data])

# print(out1.shape)
# print(out1)

# print(out2.shape)
# print(out2)

# # print(out3.shape)
# # print(out3)

# # print(out4.shape)
# # print(out4)
# pdb.set_trace()

def _batch_matmul(a, b):
    if a.get_shape()[0].value == 1:
      return tf.matmul(b, a[0, :, :], transpose_a=False, transpose_b=True)
    else:
      # return tf.batch_matmul(a, tf.expand_dims(b, 2))[:, :, 0]
      return tf.matmul(a, tf.expand_dims(b, 2))[:, :, 0]

def householder_matrix(n, k):
	H_mat = np.eye(n)
	u_var = np.random.randn(k)
	if k == 1:
		H_ref = 1
	else:
		H_ref = np.eye(k) - (2/np.dot(u_var, u_var))*np.outer(u_var, u_var)
	H_mat[n-k:, n-k:] = H_ref
	return H_mat

def householder_rotations(n, k_start=1):
	M_k_start_list = []
	W = None 
	for k in range(k_start, n+1):
		H_k = householder_matrix(n, k)
		M_k_start_list.append(H_k)
		if W is None: W = H_k
		else: W = np.matmul(H_k, W)
	return W

def householder_matrix_tf(batch, n, k, init_reflection=1, u_var=None):
	if k == 1:
		H_ref = np.sign(init_reflection)*tf.tile(tf.eye(k)[np.newaxis, :, :], [batch, 1, 1])
	else:
		if u_var is None: u_var = tf.random_normal((batch, k, 1), 0, 1, dtype=tf.float32)
		else: u_var = u_var[:, :, np.newaxis]
		H_ref = tf.eye(k)[np.newaxis, :, :] - (2./tf.matmul(u_var, u_var, transpose_a=True))*tf.matmul(u_var, u_var, transpose_b=True)
	
	H_mat_id = tf.tile(tf.concat([tf.eye(n-k)[np.newaxis, :, :], tf.zeros(shape=(k, n-k))[np.newaxis, :, :]], axis=1), [batch, 1, 1])		
	H_ref_column = tf.concat([tf.tile(tf.zeros(shape=(n-k, k))[np.newaxis, :, :], [batch, 1, 1]), H_ref], axis=1)
	H_mat = tf.concat([H_mat_id, H_ref_column], axis=2)
	return H_mat

def householder_rotations_tf(n, batch=10, k_start=1, init_reflection=1, params=None):
	M_k_start_list = []
	W = None 
	if params is not None: 
		params_split = tf.split(params, list(range(max(2, k_start), n+1)), axis=1)
		batch = tf.shape(params)[0]
	for k in range(k_start, n+1):
		if params is None or k == 1: u_var = None
		else: u_var = params_split[k-max(2, k_start)]
		H_k = householder_matrix_tf(batch, n, k, init_reflection, u_var)
		M_k_start_list.append(H_k)
		if W is None: W = H_k
		else: W = tf.matmul(H_k, W)
	return W

def tf_resize_image(x, resize_ratios=[2,2]):
	return tf.image.resize_images(x, [resize_ratios[0]*x.get_shape().as_list()[1], resize_ratios[1]*x.get_shape().as_list()[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

def tf_center_crop_image(x, resize_ratios=[28,28]):
	shape_0 = x.get_shape().as_list()[1]
	shape_1 = x.get_shape().as_list()[2]
	start_0 = int((shape_0-resize_ratios[0])/2)
	start_1 = int((shape_1-resize_ratios[1])/2)
	return x[:, start_0:start_0+resize_ratios[0], start_1:start_1+resize_ratios[1], :]


def normalized_bell_np(x):
	return 2./(1+np.exp(-8*x-4))+2./(1+np.exp(8*x-4))-3

def tf_jacobian_1(y, x):
	y_flat = tf.reshape(y, (-1,))
	jacobian_flat = tf.stack([tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flat)])
	jacobian = tf.reshape(jacobian_flat, y.shape.concatenate(x.shape))
	return tf.reshape(jacobian, [*y.get_shape().as_list(), -1, *x.get_shape().as_list()[1:]])

def tf_jacobian(y, x): # the fast one
	y_flat = tf.reshape(y, [-1])
	n = y_flat.shape[0]
	loop_vars = [tf.constant(0, tf.int32), tf.TensorArray(tf.float32, size=n)]
	aa = lambda j, _: j < n
	bb = lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x)))
	_, jacobian = tf.while_loop(aa, bb, loop_vars)
	jacobian = jacobian.stack()
	return tf.reshape(jacobian, [*y.get_shape().as_list(), -1, *x.get_shape().as_list()[1:]])

def tf_batchify_across_dim(list_of_data, axis=0):
	list_of_tensors_flat = [e['flat'] for e in list_of_data] 
	list_of_tensors_image = [e['image'] for e in list_of_data] 
	output = {'flat': None, 'image': None}
	if None not in list_of_tensors_flat: 
		list_of_sizes = [e.get_shape().as_list()[axis] for e in list_of_tensors_flat] 
		if None in list_of_sizes:
			list_of_sizes = [tf.shape(e)[axis] for e in list_of_tensors_flat] 
		output['flat'] = tf.concat(list_of_tensors_flat, axis=axis)
	else:
		list_of_sizes = [e.get_shape().as_list()[axis] for e in list_of_tensors_image] 
		if None in list_of_sizes:
			list_of_sizes = [tf.shape(e)[axis] for e in list_of_tensors_image] 
		output['image'] = tf.concat(list_of_tensors_image, axis=axis)

	return output, {'axis': axis, 'list_of_sizes': list_of_sizes}

def tf_debatchify_across_dim(concatenated, axis_sizes):
	if type(concatenated) is not dict:
		list_of_data = split_tensor_tf(concatenated, axis_sizes['axis'], axis_sizes['list_of_sizes'])		
	else:
		if concatenated['flat'] is not None:
			list_of_tensors_flat = split_tensor_tf(concatenated['flat'], axis_sizes['axis'], axis_sizes['list_of_sizes']) 
			list_of_data = [{'image': None, 'flat': e} for e in list_of_tensors_flat]
		else:
			list_of_tensors_image = split_tensor_tf(concatenated['image'], axis_sizes['axis'], axis_sizes['list_of_sizes']) 
			list_of_data = [{'image': e, 'flat': None} for e in list_of_tensors_image]
	return list_of_data

def tf_jacobian_3(y, x):
	y_list = tf.unstack(tf.reshape(y, [-1]))
	jacobian_list = [tf.gradients(y_, x)[0] for y_ in y_list]  # list [grad(y0, x), grad(y1, x), ...]
	jacobian = tf.stack(jacobian_list)
	return tf.reshape(jacobian, [*y.get_shape().as_list(), -1, *x.get_shape().as_list()[1:]])

def tf_batch_reduced_jacobian(y, x):
	batch_jacobian = tf_jacobian(tf.reduce_sum(y, axis=[0], keep_dims=False), x)
	return tf.transpose(batch_jacobian, perm=[1, 0, 2])

def triangular_matrix_mask(output_struct, input_struct):
	mask = np.greater_equal(output_struct[:, np.newaxis]-input_struct[np.newaxis, :], 0).astype(np.float32)
	return mask

def get_mask_list_for_MADE(input_dim, layer_expansions, add_mu_log_sigma_layer=False, b_normalization=True):
	layer_expansions = [1, *layer_expansions]
	layer_structures = []
	for e in layer_expansions:
		# np.random.shuffle(np.repeat(np.arange(1, input_dim+1), e))
		layer_structures.append(np.repeat(np.arange(1, input_dim+1), e))
	if add_mu_log_sigma_layer:
		layer_structures.append(np.repeat(np.arange(1, input_dim+1)[:,np.newaxis], 2, axis=1).T.flatten())
	masks = []
	for l in range(len(layer_structures)-1):
		masks.append(triangular_matrix_mask(layer_structures[l+1], layer_structures[l])[np.newaxis, :, :])
	if b_normalization:
		for i in range(len(masks)):
			# normalize rows
			# masks[i] = (masks[i].shape[2])*masks[i]/masks[i].sum(2)[:,:,np.newaxis] 
			# masks[i] = masks[i]/np.sqrt(masks[i].sum(2))[:,:,np.newaxis] 
			masks[i] = np.sqrt(masks[i].shape[2])*masks[i]/np.sqrt(masks[i].sum(2))[:,:,np.newaxis] 
			# masks[i] = masks[i]/np.sqrt(masks[i].sum(2))[:,:,np.newaxis] 
	return masks

def tf_get_mask_list_for_MADE(input_dim, layer_expansions, add_mu_log_sigma_layer=False, b_normalization=True):
	masks_np = get_mask_list_for_MADE(input_dim, layer_expansions, add_mu_log_sigma_layer=add_mu_log_sigma_layer, b_normalization=b_normalization)
	masks_tf = []
	for mask_np in masks_np:
		masks_tf.append(tf.py_func(lambda x: x, [mask_np.astype(np.float32),], [tf.float32], name="masks_list", stateful=False)[0])
	return masks_tf

def visualize_images_from_dists(sess, input_dict, batch, inference_sample, generative_sample, save_dir = '.', postfix = ''):
	sample = batch['observed']['data']
	sample_properties = batch['observed']['properties']
	for obs_type in ['flat', 'image']:
		if obs_type=='image' and sample[obs_type] is not None:
			real_sample_np = sample[obs_type]
			inference_sample_np = sess.run(inference_sample[obs_type], feed_dict = input_dict)
			generative_sample_np = sess.run(generative_sample[obs_type], feed_dict = input_dict)
			
			samples_params_np = np.array([np.array([]), real_sample_np, inference_sample_np, generative_sample_np])[1:]
			samples_params_np_interleaved = interleave_data(samples_params_np)
			visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type)

			# MNIST
			image_min = 0
			image_max = 1

			inference_sample_np_clipped = np.clip(inference_sample_np, image_min, image_max)
			generative_sample_np_clipped = np.clip(generative_sample_np, image_min, image_max)
			samples_params_np = np.array([np.array([]), real_sample_np, inference_sample_np_clipped, generative_sample_np_clipped])[1:]
			samples_params_np_interleaved = interleave_data(samples_params_np)
			visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type+'_2')

			# REAL IMAGES
			image_min = -1
			image_max = 1

			inference_sample_np_clipped = np.clip(inference_sample_np, image_min, image_max)
			generative_sample_np_clipped = np.clip(generative_sample_np, image_min, image_max)
			samples_params_np = np.array([np.array([]), real_sample_np, inference_sample_np_clipped, generative_sample_np_clipped])[1:]
			samples_params_np_interleaved = interleave_data(samples_params_np)
			visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type+'_3')

# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2
def clipped_optimizer_minimize(optimizer, loss, global_step=None, var_list=None,
							   gate_gradients=GATE_OP, aggregation_method=None,
							   colocate_gradients_with_ops=False, name=None,
							   grad_loss=None, clip_param=None):

	grads_and_vars = optimizer.compute_gradients(
		loss, var_list=var_list, gate_gradients=gate_gradients,
		aggregation_method=aggregation_method,
		colocate_gradients_with_ops=colocate_gradients_with_ops,
		grad_loss=grad_loss)
	
	if clip_param is not None and clip_param>0:
		clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_param), var) if grad is not None else (grad, var) for grad, var in grads_and_vars]
	elif clip_param<0: pdb.set_trace()
	else: clipped_grads_and_vars = grads_and_vars
	
	vars_with_grad = [v for g, v in clipped_grads_and_vars if g is not None]

	if not vars_with_grad:
		raise ValueError(
			"No gradients provided for any variable, check your graph for ops"
			" that do not support gradients, between variables %s and loss %s." %
			([str(v) for _, v in clipped_grads_and_vars], loss))

	return optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step, name=name)






# print(get_mask_list_for_MADE(3, [2,4], add_mu_sigma_layer=True))


# n = 4
# print(normalized_bell_np(np.arange(-1,1+1/n,1/n)))

# dim = 5
# batch = 4 
# k_start = 5
# params = tf.random_normal((batch, sum(list(range(max(2, k_start), dim+1)))), 0, 1, dtype=tf.float32)
# # params = tf.ones((batch, sum(list(range(max(2, k_start), dim+1)))))
# # params = None

# rot_matrix_tf = householder_rotations_tf(n=dim, k_start=k_start, init_reflection=-1, params=params)
# rot_matrix_tf2 = householder_rotations_tf(n=dim, k_start=k_start, init_reflection=1, params=params)
# rot_matrix_tf_a = tf.matmul(rot_matrix_tf, rot_matrix_tf, transpose_a=True)
# rot_matrix_tf_b = tf.matmul(rot_matrix_tf, rot_matrix_tf, transpose_a=True)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# out1, out2, out3, out4 = sess.run([rot_matrix_tf, rot_matrix_tf2, rot_matrix_tf_a, rot_matrix_tf_b])

# print(out1.shape)
# print(out1)
# print(out2.shape)
# print(out2)
# print(out3.shape)
# print(out3)
# print(out4.shape)
# print(out4)
# pdb.set_trace()


# end_vectors = []
# end_vectors2= []
# start_vector = np.asarray([[1., 0., 0.]]).T
# for i in range(out1.shape[0]):
# 	rot_matrix = out1[i]
# 	rot_matrix2 = out2[i]
# 	end_vectors.append(np.matmul(rot_matrix, start_vector))
# 	end_vectors2.append(np.matmul(rot_matrix2, start_vector))
# dataset = np.concatenate(end_vectors, axis=1).T
# dataset2 = np.concatenate(end_vectors2, axis=1).T

# dataset_plotter([dataset,], show_also=True)
# pdb.set_trace()
# dataset_plotter([dataset2,], show_also=True)
# pdb.set_trace()
# dataset_plotter([dataset, dataset2], show_also=True)
# pdb.set_trace()


# block_size = [1, 1]
# full_size = [3, 3]
# block_triangular = block_triangular_ones(full_size, block_size, trilmode=-1)
# block_diagonal = block_diagonal_ones(full_size, block_size)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# out1, out2 = sess.run([block_triangular, block_diagonal])
# print(out1)
# print(out2)
# pdb.set_trace()



# sys.stdout = helper.PrintSnooper(sys.stdout) # put it to the beginning of script

# print('postload')
# for var in tf.trainable_variables():
#     print('var', var.name, var.get_shape(), sess.run(tf.reduce_sum(var ** 2)))




# def save_checkpoint(saver, sess, global_step, exp_dir):
# 	if not os.path.exists(exp_dir): os.makedirs(exp_dir)
# 	# checkpoint_file = os.path.join(exp_dir , 'model.ckpt')
# 	# saver.save(sess, exp_dir+'/model' global_step=global_step)
# 	# saver.export_meta_graph(os.path.join(exp_dir , 'model.meta'))
# 	saver.save(sess, exp_dir+'/model', global_step=global_step)
# 	saver.export_meta_graph(exp_dir+'/model.meta')


# def load_checkpoint(saver, sess, exp_dir):	
# 	# saver = tf.train.import_meta_graph(os.path.join(exp_dir , 'model.meta'))
# 	saver.restore(sess, exp_dir+'model')

# 	# ckpt = tf.train.get_checkpoint_state(exp_dir+'/checkpoints')
# 	# if ckpt and ckpt.model_checkpoint_path:
# 	# 	saver.restore(sess, ckpt.model_checkpoint_path)
# 	# 	print('Loaded checkpoint from file: ' + ckpt.model_checkpoint_path)
# 	# else: print('No Checkpoint..')



# def network(x, y, alpha):
# 	image_shape = (28, 28, 1)
# 	x_flat = tf.reshape(x, [-1, 1, 784]) 
# 	y_flat = tf.reshape(y, [-1, 1, 784]) 
# 	concat_input = tf.concat([x_flat, y_flat, alpha], axis=-1)
# 	n_output_size = 784
# 	concat_input_flat = tf.reshape(concat_input, [-1,  concat_input.get_shape().as_list()[-1]])
# 	pdb.set_trace()

# 	lay1_flat = tf.layers.dense(inputs = concat_input_flat, units = self.config['n_decoder'], activation = activation_function)
# 	# lay1_flat = tf.concat([lay1_flat, tf.reshape(alpha, [-1,  alpha.get_shape().as_list()[-1]])], axis=-1)
# 	lay2_flat = tf.layers.dense(inputs = lay1_flat, units = self.config['n_decoder'], activation = activation_function)
# 	# lay2_flat = tf.concat([lay2_flat, tf.reshape(alpha, [-1,  alpha.get_shape().as_list()[-1]])], axis=-1)
# 	lay3_flat = tf.layers.dense(inputs = lay2_flat, units = self.config['n_decoder'], activation = activation_function)
# 	# lay3_flat = tf.concat([lay3_flat, tf.reshape(alpha, [-1,  alpha.get_shape().as_list()[-1]])], axis=-1)
# 	lay5_flat = tf.layers.dense(inputs = lay1_flat, units = n_output_size, activation = None)
# 	out = tf.reshape(lay5_flat, [-1, *x.get_shape().as_list()[1:]])
	


# N = 1
# batch_size = 30
# with tf.Graph().as_default():
# 	x = tf.placeholder(tf.float32, shape=[None, N])
# 	spur_1 = tf.placeholder(tf.float32, shape=[None, 784])
# 	spur_2 = tf.placeholder(tf.float32, shape=[None, 784])
# 	input_x = tf.concat([x, spur], axis=-1)
# 	h = tf.layers.dense(inputs = input_x, units = 784, activation = tf.nn.sigmoid)
# 	y = tf.layers.dense(inputs = h, units = 784, activation = tf.nn.sigmoid)
# 	vv = tf.gradients(y, x)
# 	jacobian = tf_batch_reduced_jacobian(y, x)

# 	init = tf.global_variables_initializer()
# 	saver = tf.train.Saver()
# 	sess = tf.InteractiveSession()
# 	sess.run(init)

# x_val = np.random.randn(batch_size, N)
# spur1_val = np.random.randn(batch_size, 784)
# spur2_val = np.random.randn(batch_size, 784)
# y_val, vv_val = sess.run([y, vv], feed_dict={x:x_val, spur:spur_val})
# start = time.time(); jacobian_val = sess.run(jacobian, feed_dict={x:x_val, spur_1:spur1_val, spur_1:spur1_val}); end = time.time(); timing= end-start
# print(jacobian_val)
# print('Timing (s),', timing)
# pdb.set_trace()








# N = 1
# batch_size = 30
# with tf.Graph().as_default():
# 	x = tf.placeholder(tf.float32, shape=[None, N])
# 	spur = tf.placeholder(tf.float32, shape=[None, 400])
# 	input_x = tf.concat([x, spur], axis=-1)
# 	h = tf.layers.dense(inputs = input_x, units = 784, activation = tf.nn.sigmoid)
# 	y = tf.layers.dense(inputs = h, units = 784, activation = tf.nn.sigmoid)
# 	vv = tf.gradients(y, x)
# 	jacobian = tf_batch_reduced_jacobian(y, x)

# 	init = tf.global_variables_initializer()
# 	saver = tf.train.Saver()
# 	sess = tf.InteractiveSession()
# 	sess.run(init)

# x_val = np.random.randn(batch_size, N)
# spur_val = np.random.randn(batch_size, 400)
# y_val, vv_val = sess.run([y, vv], feed_dict={x:x_val, spur:spur_val})
# start = time.time(); 
# for i in range(100):
# 	jacobian_val = sess.run(jacobian, feed_dict={x:x_val, spur:spur_val}); 
# end = time.time(); timing= end-start
# print(jacobian_val)
# print('Timing (s),', timing)
# pdb.set_trace()




# jacobian1 = tf_jacobian(tf.reduce_sum(y, axis=[0], keep_dims=True), x)
# jacobian2 = tf_jacobian(tf.reduce_sum(y, axis=[0], keep_dims=False), x)
# jacobian3 = tf_jacobian(y, x)

# start = time.time(); jacobian_val1 = sess.run(jacobian1, feed_dict={x:x_val}); end = time.time(); timing1= end-start
# start = time.time(); jacobian_val2 = sess.run(jacobian2, feed_dict={x:x_val}); end = time.time(); timing2= end-start
# start = time.time(); jacobian_val3 = sess.run(jacobian3, feed_dict={x:x_val}); end = time.time(); timing3= end-start
# print('Timing (s),', timing1)
# print('Timing (s),', timing2)
# print('Timing (s),', timing3)
# print(jacobian_val1)
# print(jacobian_val2)
# print(jacobian_val3)
# nonzero_jacobian = np.concatenate([jacobian_val3[0,:,0,:][np.newaxis,:,:], jacobian_val3[1,:,1,:][np.newaxis,:,:], jacobian_val3[2,:,2,:][np.newaxis,:,:]], axis=0)
# print('Different?', np.any(np.abs(jacobian_val2.transpose([1,0,2])-nonzero_jacobian)>0))
# print('Different?', np.any(np.abs(jacobian_val1-jacobian_val3)>0))
# print('Different?', np.any(np.abs(jacobian_val2-jacobian_val3)>0))

