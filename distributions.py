import pdb
import numpy as np
import math
import helper
import tensorflow as tf

####  DIVERGENCES

class FDivTotalVariation():
	def __init__(self, name = '/FDivTotalVariation'):
		self.name = name

	def generator_function(self, u):
		return 0.5*tf.abs(u-1)

	def conjugate_function(self, t):
		return t

	def codomain_function(self, v):
		return 0.5*tf.nn.tanh(v)

class FDivKL():
	def __init__(self, name = '/FDivKL'):
		self.name = name

	def generator_function(self, u):
		return u*tf.log(u)

	def conjugate_function(self, t):
		return tf.exp(t-1)

	def codomain_function(self, v):
		return v

class FDivReverseKL():
	def __init__(self, name = '/FDivReverseKL'):
		self.name = name

	def generator_function(self, u):
		return -tf.log(u)

	def conjugate_function(self, t):
		return -1-tf.log(-t)

	def codomain_function(self, v):
		return -tf.exp(-v)

class FDivPearsonChiSquared():
	def __init__(self, name = '/FDivPearsonChiSquared'):
		self.name = name
	
	def generator_function(self, u):
		return (u-1)**2

	def conjugate_function(self, t):
		return 0.25*(t**2)+t

	def codomain_function(self, v):
		return v

class FDivNeymanChiSquared():
	def __init__(self, name = '/FDivNeymanChiSquared'):
		self.name = name
	
	def generator_function(self, u):
		return ((1-u)**2)/(u+1e-7)

	def conjugate_function(self, t):
		return 2-2*tf.sqrt(1-t)

	def codomain_function(self, v):
		return 1-tf.exp(-v)

class FDivSquaredHellinger():
	def __init__(self, name = '/FDivSquaredHellinger'):
		self.name = name
	
	def generator_function(self, u):
		return (tf.sqrt(u)-1)**2

	def conjugate_function(self, t):
		return t/(1-t)

	def codomain_function(self, v):
		return 1-tf.exp(-v)

class FDivJS():
	def __init__(self, name = '/FDivJS'):
		self.name = name
	
	def generator_function(self, u):
		return -(u+1)*tf.log((u+1)/2)+u*tf.log(u)

	def conjugate_function(self, t):
		return -tf.log(2-tf.exp(t))

	def codomain_function(self, v):
		return np.log(2)-tf.log(1+tf.exp(-v))

class FDivAdHocGANJS():
	def __init__(self, name = '/FDivAdHocGANJS'):
		self.name = name
	
	def generator_function(self, u):
		return -(u+1)*tf.log((u+1))+u*tf.log(u)

	def conjugate_function(self, t):
		return -tf.log(1-tf.exp(t))

	def codomain_function(self, v):
		return -tf.log(1+tf.exp(-v))

class KLDivDiagGaussianVsDiagGaussian():
	def __init__(self, name = 'KLDivDiagGaussianVsDiagGaussian'):
		self.name = name

	def forward(self, dist1, dist2):
		assert(type(dist1) == DiagonalGaussianDistribution)
		assert(type(dist2) == DiagonalGaussianDistribution)
		mean_diff_sq = (dist2.mean - dist1.mean)**2
		log_var1 = 2*dist1.log_std
		log_var2 = 2*dist2.log_std
		scale_factor = tf.exp(-log_var2)
		first = 2*tf.reduce_sum(dist2.log_std-dist1.log_std, axis=1, keep_dims=True)
		sec_third = tf.reduce_sum(tf.exp(log_var1-log_var2), axis=1, keep_dims=True)-log_var1.get_shape().as_list()[1]
		fourth = tf.reduce_sum((mean_diff_sq*scale_factor), axis=1, keep_dims=True)
		KLD_sample = 0.5*(first+sec_third+fourth)
		return KLD_sample

class KLDivDiagGaussianVsNormal():
	def __init__(self, name = 'KLDivDiagGaussianVsNormal'):
		self.name = name

	def forward(self, dist):
		assert(type(dist) == DiagonalGaussianDistribution)
		log_var = dist.log_std*2
		KLD_element = -0.5*((log_var+1)-((dist.mean**2)+(tf.exp(log_var))))
		KLD_sample = tf.reduce_sum(KLD_element, axis=1, keep_dims=True)
		return KLD_sample 

####  UNIFORM DISTRIBUTION

class UniformDistribution():
	def __init__(self, params=None, shape=None, name='UniformDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			self.low = params[:, :int(params.get_shape().as_list()[1]/2.)]
			self.high = params[:, int(params.get_shape().as_list()[1]/2.):]
		self.name = name

	def num_params(num_dim):
		return 2*num_dim

	def get_interpretable_params(self):
		return [self.low, self.high]

	def sample(self, b_mode=False):
		if b_mode: sample = (self.low+self.high)/2.
		else: sample = tf.random_uniform(tf.shape(self.low), 0, 1, dtype=tf.float32)*(self.high-self.low)+self.low
		return sample

	def log_pdf(self, sample):
		return 1./(self.high-self.low+1e-7)


####  BERNOULLI DISTRIBUTION

class BernoulliDistribution():
	def __init__(self, params=None, shape=None, b_sigmoid=True, name='BernoulliDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			if b_sigmoid:
				self.mean = tf.nn.sigmoid(params)
			else:
				self.mean = params				
		else: print('Invalid Option. BernoulliDistribution.'); quit()
		self.name = name
		# self._dist = tf.distributions.Bernoulli(probs=self.mean)

	def num_params(num_dim):
		return num_dim

	def get_interpretable_params(self):
		return [self.mean]

	def sample(self, b_mode=False):
		if b_mode: sample = self.mean #sample = tf.stop_gradient(tf.cast(self.mean > 0.5, tf.float32))
		else: sample = tf.stop_gradient(tf.cast(tf.random_uniform(shape=(tf.shape(self.mean)[0], 1)) < self.mean, tf.float32))
		return sample

	def log_pdf(self, sample):
		assert(len(sample.get_shape())==2)
		# sample = tf.reshape(sample, [tf.shape(sample)[0], -1])
		binary_cross_entropy_dims = sample*tf.log(1e-7+self.mean)+ (1.0-sample)*tf.log(1e-7+1.0-self.mean)
		log_prob = tf.reduce_sum(binary_cross_entropy_dims, axis=1, keep_dims=True)
		
		# log_prob = tf.reduce_sum(self._dist._log_prob(sample), axis=1, keep_dims=True)
		return log_prob


####  CATEGORICAL/Boltzman DISTRIBUTION

class BoltzmanDistribution():
	def __init__(self, params=None, temperature=1, shape=None, name='BoltzmanDistribution'):
		if len(params.get_shape().as_list()) == 2: self.logits = params
		else: print('Invalid Option. BoltzmanDistribution.'); quit()
		self.temperature = temperature
		self.name = name

	@staticmethod
	def num_params(num_dim):
		return num_dim

	def get_interpretable_params(self):
		return [tf.nn.softmax(self.logits/self.temperature)]

	def sample(self, b_mode=False):
		prob_w = tf.nn.softmax(self.logits/self.temperature)
		if b_mode: 
			indices = tf.argmax(prob_w, axis=1)
		else:
			sed = tf.random_uniform(shape=(prob_w.get_shape().as_list()[0], 1))
			prob_w_accum = tf.cumsum(prob_w, axis=1)
			indices = tf.argmax(tf.cast(prob_w_accum>sed, tf.float32), axis=1)
		sample = tf.stop_gradient(tf.cast(tf.one_hot(indices, depth=prob_w.get_shape().as_list()[1], on_value=1, off_value=0, axis=1), tf.float32))
		return sample

	def entropy(self):
		log_prob_all = tf.nn.log_softmax(self.logits/self.temperature)
		prob_all = tf.exp(log_prob_all)
		entropy = tf.reduce_sum(-log_prob_all*prob_all, axis=1, keep_dims=True)
		return entropy

	def log_pdf(self, sample_one_hot):
		sample_one_hot_flat = tf.reshape(sample_one_hot, [sample_one_hot.get_shape().as_list()[0], -1])
		log_prob_all = tf.nn.log_softmax(self.logits/self.temperature)
		log_prob = tf.reduce_sum(log_prob_all*sample_one_hot_flat, axis=1, keep_dims=True)
		return log_prob
	
	
####  DIAGONAL GAUSSIAN DISTRIBUTION

class DiagonalGaussianDistribution():
	def __init__(self, params=None, shape = None, name = 'DiagonalGaussianDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			self.mean = params[:, :int(params.get_shape().as_list()[1]/2.)]
			self.log_std = params[:, int(params.get_shape().as_list()[1]/2.):]
		else: print('Invalid Option. DiagonalGaussianDistribution.'); quit()
		self.name = name
		# self._dist = tf.distributions.Normal(loc=self.mean, scale=tf.exp(self.log_std))

	@staticmethod
	def num_params(num_dim):
		return 2*num_dim
		
	def get_interpretable_params(self):
		return [self.mean, tf.exp(self.log_std)]

	def sample(self, b_mode=False):
		if b_mode: sample = self.mean
		else:
			eps = tf.random_normal(shape=tf.shape(self.log_std))
			sample = (tf.exp(self.log_std)*eps)+self.mean
		return sample

	def log_pdf(self, sample):
		assert(len(sample.get_shape())==2)
		# sample = tf.reshape(sample, [tf.shape(sample)[0], -1])
		log_var = self.log_std*2
		unnormalized_log_prob = -0.5*tf.reduce_sum(((self.mean-sample)**2)/(tf.exp(log_var)), axis=1, keep_dims=True)
		log_partition = -0.5*tf.reduce_sum(log_var, axis=1, keep_dims=True)+math.log(2*math.pi)*(-self.mean.get_shape().as_list()[1]/2.0)
		log_prob = unnormalized_log_prob+log_partition

		# log_prob = tf.reduce_sum(self._dist._log_prob(sample), axis=1, keep_dims=True)
		return log_prob

# ####  DIAGONAL BETA DISTRIBUTION

# class DiagonalBetaDistribution():
# 	def __init__(self, params=None, shape = None, name = 'DiagonalBetaDistribution'):
# 		if len(params.get_shape().as_list()) == 2: 
# 			self.alpha = params[:, :int(params.get_shape().as_list()[1]/2.)]
# 			self.beta = params[:, int(params.get_shape().as_list()[1]/2.):]
# 			pdb.set_trace()
# 		else: print('Invalid Option. DiagonalBetaDistribution.'); quit()
# 		self.name = name

# 	@staticmethod
# 	def num_params(num_dim):
# 		return 2*num_dim
		
# 	def get_interpretable_params(self):
# 		return [self.mean, tf.exp(self.log_std)]

# 	def sample(self, b_mode=False):
# 		if b_mode: 
# 			sample = (self.alpha-1)/(self.alpha+self.beta-2)
# 		else:
# 			eps = tf.random_normal(shape=tf.shape(self.log_std))
# 			sample = (tf.exp(self.log_std)*eps)+self.mean
# 		return sample

# 	def log_pdf(self, sample):
# 		sample = tf.reshape(sample, [tf.shape(sample)[0], -1])
# 		log_var = self.log_std*2
# 		unnormalized_log_prob = -0.5*tf.reduce_sum(((self.mean-sample)**2)/(tf.exp(log_var)), axis=1, keep_dims=True)
# 		log_partition = -0.5*tf.reduce_sum(log_var, axis=1, keep_dims=True)+math.log(2*math.pi)*(-self.mean.get_shape().as_list()[1]/2.0)
# 		log_prob = unnormalized_log_prob+log_partition
# 		return log_prob


####  BERNOULLI DISTRIBUTION

class DiracDistribution():
	def __init__(self, params=None, shape=None, name='DiracDistribution'):
		if len(params.get_shape().as_list()) == 2: self.mean = params
		else: print('Invalid Option. DiracDistribution.'); quit()
		self.name = name

	def num_params(num_dim):
		return num_dim

	def get_interpretable_params(self):
		return [self.mean]

	def sample(self, b_mode=False):
		if b_mode: sample = self.mean
		else: sample = self.mean
		return sample

	def log_pdf(self, sample):
		print('Requested pdf of delta sample.')
		return None



####  PRODUCT DISTRIBUTION

class ProductDistribution():
	def __init__(self, sample_properties = None, params = None, name = 'ProductDistribution'):
		self.name = name
		self.sample_properties = sample_properties

		self.dist_class_list = {'flat': [DistributionsAKA[e['dist']] for e in self.sample_properties['flat']], 
								'image': [DistributionsAKA[e['dist']] for e in self.sample_properties['image']]}

		self.param_sizes = {'flat': [self.dist_class_list['flat'][i].num_params(np.prod(e['size'][2:])) for i, e in enumerate(self.sample_properties['flat'])], 
							'image': [self.dist_class_list['image'][i].num_params(np.prod(e['size'][2:])) for i, e in enumerate(self.sample_properties['image'])]}

		self.params = {'flat': helper.split_tensor_tf(params['flat'], -1, [int(self.param_sizes['flat'][i]/np.prod(e['size'][2:-1])) for i, e in enumerate(self.sample_properties['flat'])]), 
					   'image':helper.split_tensor_tf(params['image'], -1, [int(self.param_sizes['image'][i]/np.prod(e['size'][2:-1])) for i, e in enumerate(self.sample_properties['image'])])} 
		
		self.params_flat = {'flat': [tf.reshape(self.params['flat'][i], [-1, np.prod(self.params['flat'][i].get_shape().as_list()[2:])]) for i, e in enumerate(self.params['flat'])],
					 		'image': [tf.reshape(self.params['image'][i], [-1, np.prod(self.params['image'][i].get_shape().as_list()[2:])]) for i, e in enumerate(self.params['image'])]}

		self.dist_list = {'flat': [dist(params = self.params_flat['flat'][i]) for i, dist in enumerate(self.dist_class_list['flat'])],
						  'image': [dist(params = self.params_flat['image'][i]) for i, dist in enumerate(self.dist_class_list['image'])]}

	def sample(self, b_mode=False):
		samples = {'flat': None, 'image': None}
		for obs_type in ['flat', 'image']:
			samples_curr = []
			for i, dist in enumerate(self.dist_list[obs_type]):
				sample = tf.reshape(dist.sample(b_mode), [-1, *self.sample_properties[obs_type][i]['size'][1:]])
				samples_curr.append(sample)
			if len(samples_curr) > 0: 
				samples[obs_type] = tf.concat(samples_curr, axis=-1)
		return samples

	def entropy(self):
		entropies_all = None
		for obs_type in ['flat', 'image']:
			if self.params[obs_type] is not None:
				entropies = []
				for i, e in enumerate(self.dist_list[obs_type]):
					entropies.append(tf.reshape(e.entropy(), [-1, self.params[obs_type][i].get_shape().as_list()[1], 1]))
				if entropies_all is None: entropies_all = entropies
				else: entropies_all = entropies_all + entropies
		return helper.list_sum(entropies_all)

	def log_pdf(self, sample):
		log_pdfs_all = None
		for obs_type in ['flat', 'image']:
			if sample[obs_type] is not None:
				split = helper.split_tensor_tf(sample[obs_type], -1, [e['size'][-1] for e in self.sample_properties[obs_type]])
				split_batched = [tf.reshape(e, [-1, *e.get_shape().as_list()[2:]]) for e in split]
				log_pdfs = []
				for i, e in enumerate(self.dist_list[obs_type]):
					split_batched_flat = tf.reshape(split_batched[i], [-1, np.prod(split_batched[i].get_shape().as_list()[1:])])
					log_pdfs.append(tf.reshape(e.log_pdf(split_batched_flat), [-1, sample[obs_type].get_shape().as_list()[1], 1]))
				if log_pdfs_all is None: log_pdfs_all = log_pdfs
				else: log_pdfs_all = log_pdfs_all + log_pdfs
		return helper.list_sum(log_pdfs_all)

####  TRANSFORMED DISTRIBUTION

class TransformedDistribution():
	"""
	Args:
		base_dist: base distribution.
		transforms_list: list of transforms
		batch_size : Batch size for distribution functions. 

	Raises:
		ValueError: 
	"""
	def __init__(self, base_dist=None, transforms = [], name='transform_dist'):
		self._base_dist = base_dist
		self._transforms = transforms
		self._log_pdf_cache = {}
		self._base_log_pdf_cache = {}

	@property
	def dim(self):
		return self._transforms[len(self._transforms)-1].output_dim

	def add_transforms(self, transforms_list):
		self._transforms = self._transforms + transforms_list

	def sample(self, base_sample=None, base_log_pdf=None):
		if base_sample is None or base_log_pdf is None:
			if self._base_dist is None:
				print('No base sample or distribution given.'); quit()
			else:
				base_sample = self._base_dist.sample()
				base_log_pdf = self._base_dist.log_pdf(base_sample)
		curr_sample, curr_log_pdf = base_sample, base_log_pdf 
		for i in range(len(self._transforms)): 
			curr_sample, curr_log_pdf = self._transforms[i].transform(curr_sample, curr_log_pdf)
		self._log_pdf_cache[curr_sample] = curr_log_pdf
		self._base_log_pdf_cache[base_sample] = base_log_pdf
		return curr_sample, base_sample

	def log_pdf(self, sample):
		if sample not in self._log_pdf_cache: 
			print('Requested log pdf of sample not generated by distribution.'); quit()
		return self._log_pdf_cache[sample]
	
	def base_log_pdf(self, sample):
		if sample not in self._base_log_pdf_cache: 
			print('Requested log pdf of base sample not generated by distribution.'); quit()
		return self._base_log_pdf_cache[sample]



####  AKA
DistributionsAKA = {}
DistributionsAKA['cat'] = BoltzmanDistribution
DistributionsAKA['bern'] = BernoulliDistribution
DistributionsAKA['cont'] = DiagonalGaussianDistribution
# DistributionsAKA['cont'] = DiagonalBetaDistribution
DistributionsAKA['dirac'] = DiracDistribution

def visualizeProductDistribution(sess, input_dict, batch, obs_dist, sample_obs_dist, save_dir = '.', postfix = ''):
	
	sample = batch['observed']['data']
	sample_properties = batch['observed']['properties']
	for obs_type in ['flat', 'image']:
		if sample[obs_type] is not None:
			
			sample_split = helper.split_tensor_np(sample[obs_type], -1, [e['size'][-1] for e in obs_dist.sample_properties[obs_type]])
			param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(obs_dist.dist_list[obs_type])]			
			rand_param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(sample_obs_dist.dist_list[obs_type])]

			param_split = sess.run(param_split_tf, feed_dict = input_dict)
			rand_param_split = sess.run(rand_param_split_tf, feed_dict = input_dict)
			samples_params_np = np.array([np.array([]), *sample_split, *param_split, *rand_param_split])[1:]

			if obs_type == 'flat':	
				cont_var_filter = np.tile(np.asarray([e['dist'] == 'cont' for e in batch['observed']['properties'][obs_type]]), 3)
				not_cont_var_filter = np.tile(np.asarray([e['dist'] != 'cont' for e in batch['observed']['properties'][obs_type]]), 3)
				if sum(not_cont_var_filter) > 0: helper.visualize_flat(samples_params_np[not_cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)
				if sum(cont_var_filter) > 0: helper.visualize_vectors(samples_params_np[cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)

			if obs_type == 'image':
				samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				helper.visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type)
			
def visualizeProductDistribution2(sess, input_dict, batch, obs_dist, save_dir = '.', postfix = ''):
	
	sample = batch['observed']['data']
	sample_properties = batch['observed']['properties']
	for obs_type in ['flat', 'image']:
		if sample[obs_type] is not None:
			
			sample_split = helper.split_tensor_np(sample[obs_type], -1, [e['size'][-1] for e in obs_dist.sample_properties[obs_type]])
			param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(obs_dist.dist_list[obs_type])]			
			
			param_split = sess.run(param_split_tf, feed_dict = input_dict)
			samples_params_np = np.array([np.array([]), *sample_split, *param_split])[1:]

			if obs_type == 'flat':	
				cont_var_filter = np.tile(np.asarray([e['dist'] == 'cont' for e in batch['observed']['properties'][obs_type]]), 2)
				not_cont_var_filter = np.tile(np.asarray([e['dist'] != 'cont' for e in batch['observed']['properties'][obs_type]]), 2)
				if sum(not_cont_var_filter) > 0: helper.visualize_flat2(samples_params_np[not_cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)
				if sum(cont_var_filter) > 0: helper.visualize_flat2(samples_params_np[cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)

			if obs_type == 'image':
				samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				helper.visualize_images(samples_params_np_interleaved, save_dir = save_dir, postfix = postfix+'_'+obs_type)

def visualizeProductDistribution3(sess, input_dict, batch, obs_dist, transport_dist, rec_dist, sample_obs_dist, save_dir = '.', postfix = ''):
	
	sample = batch['observed']['data']
	sample_properties = batch['observed']['properties']
	for obs_type in ['flat', 'image']:
		if sample[obs_type] is not None:

			sample_split = helper.split_tensor_np(sample[obs_type], -1, [e['size'][-1] for e in obs_dist.sample_properties[obs_type]])
			param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(obs_dist.dist_list[obs_type])]			
			transport_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(transport_dist.dist_list[obs_type])]			
			rec_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(rec_dist.dist_list[obs_type])]			
			rand_param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(sample_obs_dist.dist_list[obs_type])]

			param_split, transport_split, rec_split, rand_param_split = sess.run([param_split_tf, transport_split_tf, rec_split_tf, rand_param_split_tf], feed_dict = input_dict)
			rand_param_split2 = None
			while rand_param_split2 is None or rand_param_split2.shape[0]<300:
				if rand_param_split2 is None: rand_param_split2 = sess.run(rand_param_split_tf, feed_dict = input_dict)[0]
				else: rand_param_split2 = np.concatenate([rand_param_split2, sess.run(rand_param_split_tf, feed_dict = input_dict)[0]], axis=0)
			samples_params_np = np.array([np.array([]), *sample_split, *transport_split, *rec_split, *param_split, *rand_param_split])[1:]
			# rand_param_split2 (300, 1, 64, 64, 3)

			if obs_type == 'flat':	
				cont_var_filter = np.tile(np.asarray([e['dist'] == 'cont' for e in batch['observed']['properties'][obs_type]]), 4)
				not_cont_var_filter = np.tile(np.asarray([e['dist'] != 'cont' for e in batch['observed']['properties'][obs_type]]), 4)
				if sum(not_cont_var_filter) > 0: helper.visualize_flat(samples_params_np[not_cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)
				if sum(cont_var_filter) > 0: helper.visualize_vectors(samples_params_np[cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)

			if obs_type == 'image':
				samples_params_np = np.array([np.array([]), *sample_split, *transport_split, *rec_split, *param_split, *rand_param_split])[1:]
				samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				helper.visualize_images2(samples_params_np_interleaved, block_size=[sample_split[0].shape[0], len(samples_params_np)], save_dir=save_dir+'normal/', postfix=postfix+'_'+obs_type+'_normal')
				helper.visualize_images2(rand_param_split2[:int(np.sqrt(rand_param_split2.shape[0]))**2, ...], block_size=[int(np.sqrt(rand_param_split2.shape[0])), int(np.sqrt(rand_param_split2.shape[0]))], save_dir=save_dir+'normal_sample_only/', postfix=postfix+'_'+obs_type+'_normal_sample_only')

				# image_min, image_max = 0, 1
				# samples_params_np = np.array([np.array([]), *sample_split, *np.clip(transport_split, image_min, image_max), *np.clip(rec_split, image_min, image_max), 
				# 					*np.clip(param_split, image_min, image_max), *np.clip(rand_param_split, image_min, image_max)])[1:]
				# samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				# helper.visualize_images2(samples_params_np_interleaved, block_size=[sample_split[0].shape[0], len(samples_params_np)], save_dir=save_dir+'zeroone/', postfix=postfix+'_'+obs_type+'_zeroone')
				# helper.visualize_images2(np.clip(rand_param_split2, image_min, image_max)[:int(np.sqrt(rand_param_split2.shape[0]))**2, ...], block_size=[int(np.sqrt(rand_param_split2.shape[0])), int(np.sqrt(rand_param_split2.shape[0]))], save_dir=save_dir+'zeroone_sample_only/', postfix=postfix+'_'+obs_type+'_zeroone_sample_only')

				# image_min, image_max = -1, 1
				# samples_params_np = np.array([np.array([]), *sample_split, *np.clip(transport_split, image_min, image_max), 
				# 							  *np.clip(param_split, image_min, image_max), *np.clip(rand_param_split, image_min, image_max)])[1:]
				# samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				# helper.visualize_images2(samples_params_np_interleaved, block_size=[sample_split[0].shape[0], len(samples_params_np)], save_dir=save_dir+'2/', postfix=postfix+'_'+obs_type+'_2')


def visualizeProductDistribution4(sess, input_dict, batch, real_dist, transport_dist, reg_target_dist, rec_dist, obs_dist, sample_obs_dist, save_dir = '.', postfix = '', postfix2 = None, b_zero_one_range=False):
	sample = batch['observed']['data']
	sample_properties = batch['observed']['properties']
	for obs_type in ['flat', 'image']:
		if sample[obs_type] is not None:

			sample_split = helper.split_tensor_np(sample[obs_type], -1, [e['size'][-1] for e in obs_dist.sample_properties[obs_type]])

			real_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(real_dist.dist_list[obs_type])]			
			transport_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(transport_dist.dist_list[obs_type])]			
			reg_target_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(reg_target_dist.dist_list[obs_type])]			
			rec_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(rec_dist.dist_list[obs_type])]			
			param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(obs_dist.dist_list[obs_type])]			
			rand_param_split_tf = [tf.reshape(e.get_interpretable_params()[0], list(sample_split[i].shape)) for i, e in enumerate(sample_obs_dist.dist_list[obs_type])]
			
			real_split, transport_split, reg_target_split, rec_split, param_split, rand_param_split = sess.run([real_split_tf, transport_split_tf, reg_target_split_tf, rec_split_tf, param_split_tf, rand_param_split_tf], feed_dict = input_dict)
			rand_param_split2 = None
			while rand_param_split2 is None or rand_param_split2.shape[0]<400:
				if rand_param_split2 is None: rand_param_split2 = sess.run(rand_param_split_tf, feed_dict = input_dict)[0]
				else: rand_param_split2 = np.concatenate([rand_param_split2, sess.run(rand_param_split_tf, feed_dict = input_dict)[0]], axis=0)
			samples_params_np = np.array([np.array([]), *sample_split, *transport_split, *reg_target_split, *rec_split, *param_split, *rand_param_split])[1:]
			# rand_param_split2 (300, 1, 64, 64, 3)

			if obs_type == 'flat':	
				cont_var_filter = np.tile(np.asarray([e['dist'] == 'cont' for e in batch['observed']['properties'][obs_type]]), 4)
				not_cont_var_filter = np.tile(np.asarray([e['dist'] != 'cont' for e in batch['observed']['properties'][obs_type]]), 4)
				if sum(not_cont_var_filter) > 0: helper.visualize_flat(samples_params_np[not_cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)
				if sum(cont_var_filter) > 0: helper.visualize_vectors(samples_params_np[cont_var_filter], save_dir = save_dir, postfix = postfix+'_'+obs_type)

			if obs_type == 'image':
				if b_zero_one_range: 
					np.clip(transport_split[0], 0, 1, out=transport_split[0])
					np.clip(reg_target_split[0], 0, 1, out=reg_target_split[0])
					np.clip(rec_split[0], 0, 1, out=rec_split[0])
					np.clip(param_split[0], 0, 1, out=param_split[0])
					np.clip(rand_param_split[0], 0, 1, out=rand_param_split[0])
					np.clip(rand_param_split2[0], 0, 1, out=rand_param_split2[0])

				samples_params_np = np.array([np.array([]), *real_split, *transport_split, *reg_target_split, *rec_split, *param_split, *rand_param_split])[1:]
				samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				helper.visualize_images2(samples_params_np_interleaved, block_size=[sample_split[0].shape[0], len(samples_params_np)], save_dir=save_dir+'_normal/', postfix='normal_'+postfix, postfix2='normal_'+postfix2)
				helper.visualize_images2(rand_param_split2[:int(np.sqrt(rand_param_split2.shape[0]))**2, ...], block_size=[int(np.sqrt(rand_param_split2.shape[0])), int(np.sqrt(rand_param_split2.shape[0]))], save_dir=save_dir+'_sample_only/', postfix='sample_only_'+postfix, postfix2='sample_only_'+postfix2)

				# image_min, image_max = 0, 1
				# samples_params_np = np.array([np.array([]), *sample_split, *np.clip(transport_split, image_min, image_max), *np.clip(rec_split, image_min, image_max), 
				# 					*np.clip(param_split, image_min, image_max), *np.clip(rand_param_split, image_min, image_max)])[1:]
				# samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				# helper.visualize_images2(samples_params_np_interleaved, block_size=[sample_split[0].shape[0], len(samples_params_np)], save_dir=save_dir+'zeroone/', postfix=postfix+'_'+obs_type+'_zeroone')
				# helper.visualize_images2(np.clip(rand_param_split2, image_min, image_max)[:int(np.sqrt(rand_param_split2.shape[0]))**2, ...], block_size=[int(np.sqrt(rand_param_split2.shape[0])), int(np.sqrt(rand_param_split2.shape[0]))], save_dir=save_dir+'zeroone_sample_only/', postfix=postfix+'_'+obs_type+'_zeroone_sample_only')

				# image_min, image_max = -1, 1
				# samples_params_np = np.array([np.array([]), *sample_split, *np.clip(transport_split, image_min, image_max), 
				# 							  *np.clip(param_split, image_min, image_max), *np.clip(rand_param_split, image_min, image_max)])[1:]
				# samples_params_np_interleaved = helper.interleave_data(samples_params_np)
				# helper.visualize_images2(samples_params_np_interleaved, block_size=[sample_split[0].shape[0], len(samples_params_np)], save_dir=save_dir+'2/', postfix=postfix+'_'+obs_type+'_2')



# print('sample_split: ', sample_split[0].min(), sample_split[0].max())
# 				print('transport_split: ', transport_split[0].min(), transport_split[0].max())
# 				print('param_split: ', param_split[0].min(), param_split[0].max())
# 				print('rand_param_split: ', rand_param_split[0].min(), rand_param_split[0].max())













				# print('This is the absolute value of the difference between the first two channels ')
				# print(np.abs(reg_target_split[0][0][0][:,:,0]-reg_target_split[0][0][0][:,:,1]).max())
				# print(np.min(reg_target_split[0][:,:,:,:,0]), np.max(reg_target_split[0][:,:,:,:,0]))
				# print(np.min(reg_target_split[0][:,:,:,:,1]), np.max(reg_target_split[0][:,:,:,:,1]))
				# print(np.min(reg_target_split[0][:,:,:,:,2]), np.max(reg_target_split[0][:,:,:,:,2]))
				
				
				# import matplotlib
				# import platform
				# if platform.dist()[0] == 'centos':
				# 	matplotlib.use('Agg')
				# elif platform.dist()[0] == 'debian': 
				# 	matplotlib.use('Agg')
				# else: 
				# 	matplotlib.use('TkAgg')
				# import matplotlib.pyplot as plt

				# # pdb.set_trace()
				# # fig = plt.figure(figsize=(15, 15), dpi=300)
				# ttv = reg_target_split[0][0][0]
				# # np.clip(ttv, 0, 1, out=ttv)
				# # plt.imshow(ttv)
				# # plt.savefig('blah.png')
				# # plt.close('all')
				# # print(ttv[12:17,12:17,2]-ttv[12:17,12:17,0])
				# print(ttv[12:17,12:17,0])
				# print(ttv[12:17,12:17,1])

				# samples_params_np = np.array([np.array([]), *real_split, *sample_split, *transport_split, *reg_target_split, *rec_split, *param_split, *rand_param_split])[1:]

