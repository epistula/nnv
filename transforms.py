"""Random variable transformation classes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import helper 
import pdb 
import numpy as np
import math 

class PlanarFlow():
    """
    Planar Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, name='planar_transform'):   
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return input_dim+input_dim+1

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        self._parameters.get_shape().assert_is_compatible_with([None, PlanarFlow.required_num_parameters(self._input_dim)])

        w = tf.slice(self._parameters, [0, 0], [-1, self._input_dim])
        u = tf.slice(self._parameters, [0, self._input_dim], [-1, self._input_dim])
        b = tf.slice(self._parameters, [0, 2*self._input_dim], [-1, 1])
        w_t_u = tf.reduce_sum(w*u, axis=[1], keep_dims=True)
        w_t_w = tf.reduce_sum(w*w, axis=[1], keep_dims=True)
        u_tilde = (u+(((tf.log(1e-7+1+tf.exp(w_t_u))-1)+w_t_u)/w_t_w)*w)

        affine = tf.reduce_sum(z0*w, axis=[1], keep_dims=True) + b
        h = tf.tanh(affine)
        z = z0+u_tilde*h

        h_prime_w = (1-tf.pow(h, 2))*w
        log_abs_det_jacobian = tf.log(1e-7+tf.abs(1+tf.reduce_sum(h_prime_w*u_tilde, axis=[1], keep_dims=True)))
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z


class RadialFlow():
    """
    Radial Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, name='radial_transform'):   
        self._parameter_scale = 1#0.1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return 1+1+input_dim

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        self._parameters.get_shape().assert_is_compatible_with([None, RadialFlow.required_num_parameters(self._input_dim)])

        alpha = tf.slice(self._parameters, [0, 0], [-1, 1])
        beta = tf.slice(self._parameters, [0, 1], [-1, 1])
        z_ref = tf.slice(self._parameters, [0, 2], [-1, self._input_dim])
        alpha_tilde = tf.log(1e-7+1+tf.exp(alpha))
        beta_tilde = tf.log(1e-7+1+tf.exp(beta)) - alpha_tilde

        z_diff = z0 - z_ref
        r = tf.sqrt(tf.reduce_sum(tf.square(z_diff), axis=[1], keep_dims=True))
        h = 1/(alpha_tilde + r)
        scale = beta_tilde * h
        z = z0 + scale * z_diff

        log_abs_det_jacobian = tf.log(1e-7+tf.abs(tf.pow(1 + scale, self._input_dim - 1) * (1 + scale * (1 - h * r))))
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z

class InverseOrderDimensionFlow():
    """
    Inverse Order Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, name='inverse_order_dimension_transform'):   
        self._input_dim = input_dim
        assert (parameters is None)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        z = tf.reverse(z0, axis=[-1,])
        log_pdf_z = log_pdf_z0
        return z, log_pdf_z

class PermuteDimensionFlow():
    """
    Permute Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, slide_to_higher=True, name='permute_dimension_transform'):   
        self._input_dim = input_dim
        self._slide_to_higher = slide_to_higher
        self._n_slide_dims = int(float(self._input_dim)/3.)+1
        assert (self._n_slide_dims != 0)
        assert (parameters is None)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        if self._slide_to_higher:
            z = tf.concat([tf.slice(z0, [0, self._n_slide_dims], [-1, self._input_dim-self._n_slide_dims]), tf.slice(z0, [0, 0], [-1, self._n_slide_dims])], axis=1)
        else:
            z = tf.concat([tf.slice(z0, [0, self._input_dim-self._n_slide_dims], [-1, self._n_slide_dims]), tf.slice(z0, [0, 0], [-1, self._input_dim-self._n_slide_dims])], axis=1)
        log_pdf_z = log_pdf_z0
        return z, log_pdf_z

class ScaleDimensionFlow():
    """
    Scale Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, scale=1/5., name='scale_dimension_transform'):   
        self._input_dim = input_dim
        self._scale = scale
        assert (parameters is None)
        # if parameters is not None: print('Parameters passed to parameterless transform.'); quit()

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        z = z0*self._scale
        
        log_abs_det_jacobian = self._input_dim*tf.log(1e-7+self._scale)
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z


class OpenIntervalDimensionFlow():
    """
    Open Interval Dimension Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    # def __init__(self, parameters, input_dim, zero_one=True, name='open_interval_dimension_transform'):  #mnist
    def __init__(self, parameters, input_dim, zero_one=False, name='open_interval_dimension_transform'):  #real
        self._input_dim = input_dim
        self._zero_one = zero_one
        assert (parameters is None)
        # if parameters is not None: print('Parameters passed to parameterless transform.'); quit()
    
    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):  
        return 0

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        if self._zero_one:
            z = tf.nn.sigmoid(z0)
            log_abs_det_jacobian = tf.reduce_sum(z0-2*tf.log(tf.exp(z0)+1), axis=[1], keep_dims=True)
            # log_abs_det_jacobian = tf.reduce_sum(log_z+tf.log(1e-7+(1-z)), axis=[1], keep_dims=True)
        else:
            z = tf.nn.tanh(z0)
            log_abs_det_jacobian = tf.reduce_sum(tf.log(1e-7+(1-z))+tf.log(1e-7+(1+z)), axis=[1], keep_dims=True)
        
        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z


class RiemannianFlow():
    """
    Projective Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, additional_dim=3, k_start=1, init_reflection=1, manifold_nonlinearity=tf.nn.tanh, polinomial_degree=3, name='riemannian_transform'):   
        self._parameter_scale = 1.
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._additional_dim = additional_dim
        self._k_start = k_start
        self._init_reflection = init_reflection
        self._polinomial_degree = polinomial_degree
        self._manifold_nonlinearity = manifold_nonlinearity
        assert(input_dim>additional_dim)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim+self._additional_dim

    def apply_manifold_nonlin(self, x_k, NonlinK_param):
        if self._manifold_nonlinearity == tf.nn.tanh: return tf.nn.tanh(x_k), helper.tanh_J(x_k)
        if self._manifold_nonlinearity == tf.nn.sigmoid: return tf.nn.sigmoid(x_k), helper.sigmoid_J(x_k)
        if self._manifold_nonlinearity == tf.nn.relu: return tf.nn.relu(x_k), helper.relu_J(x_k)
        if self._manifold_nonlinearity == helper.parametric_relu:
            param_index = 0
            positive, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)
            negative, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)
            return helper.parametric_relu(x_k, positive, negative), helper.parametric_relu_J(x_k, positive, negative)
        if self._manifold_nonlinearity == helper.polinomial_nonlin:
            param_index = 0
            positive, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)
            negative, param_index = helper.slice_parameters(NonlinK_param, param_index, self._additional_dim)            
            pdb.set_trace()
            return y_k, manifold_nonlinearity_J

    def get_rotation_tensor(self, dim, rot_params):
        rot_tensor = helper.householder_rotations_tf(n=dim, batch=tf.shape(rot_params)[0], k_start=self._k_start, init_reflection=self._init_reflection, params=rot_params)
        return rot_tensor

    @staticmethod
    def required_num_parameters(input_dim, additional_dim=3, k_start=1, manifold_nonlinearity=tf.nn.tanh, polinomial_degree=3): 
        if additional_dim == 1:
            n_C_param = input_dim
            n_RK_1_param = 1
            n_RK_2_param = 1
        else:
            n_C_param = HouseholdRotationFlow.required_num_parameters(input_dim, k_start)
            n_RK_1_param = HouseholdRotationFlow.required_num_parameters(additional_dim, k_start)
            n_RK_2_param = HouseholdRotationFlow.required_num_parameters(additional_dim, k_start)
            
        n_pre_bias_param = additional_dim
        n_pre_scale_param = additional_dim
        if manifold_nonlinearity == tf.nn.tanh or manifold_nonlinearity == tf.nn.sigmoid or manifold_nonlinearity == tf.nn.relu :
            n_NonlinK_param = 0
        elif manifold_nonlinearity == helpe.parametric_relu:
            n_NonlinK_param = 2*additional_dim
        elif manifold_nonlinearity == RiemannianFlow.polinomial_nonlin:
            n_NonlinK_param = (polinomial_degree+1)*additional_dim
        n_post_bias_param = additional_dim
        n_post_scale_param = additional_dim

        n_RN_param = HouseholdRotationFlow.required_num_parameters(input_dim, k_start)
        n_RG_param = HouseholdRotationFlow.required_num_parameters(input_dim+additional_dim, k_start)
        
        return n_C_param+n_RK_1_param+n_RK_2_param+ \
               n_pre_bias_param+n_pre_scale_param+n_NonlinK_param+ \
               n_post_bias_param+n_post_scale_param+ \
               n_RN_param+n_RG_param

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        self._parameters.get_shape().assert_is_compatible_with([None, RiemannianFlow.required_num_parameters(
            self._input_dim, self._additional_dim, self._k_start, self._manifold_nonlinearity, self._polinomial_degree)])

        param_index = 0
        if self._additional_dim == 1:             
            C_param, param_index = helper.slice_parameters(self._parameters, param_index, self._input_dim)
            RK_1_param, param_index = helper.slice_parameters(self._parameters, param_index, 1)
            RK_2_param, param_index = helper.slice_parameters(self._parameters, param_index, 1)
        else: 
            C_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._input_dim, self._k_start))
            RK_1_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._additional_dim, self._k_start))
            RK_2_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._additional_dim, self._k_start))

        pre_bias, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
        pre_scale, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
        if self._manifold_nonlinearity == tf.nn.tanh or self._manifold_nonlinearity == tf.nn.sigmoid or self._manifold_nonlinearity == tf.nn.relu :
            NonlinK_param = None
        elif manifold_nonlinearity == helper.parametric_relu:
            NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, 2*self._additional_dim)
        elif manifold_nonlinearity == helper.polinomial_nonlin:
            NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, (self._polinomial_degree+1)*self._additional_dim)   
        post_bias, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
        post_scale, param_index = helper.slice_parameters(self._parameters, param_index, self._additional_dim)
        
        RN_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._input_dim, self._k_start))
        RG_param, param_index = helper.slice_parameters(self._parameters, param_index, HouseholdRotationFlow.required_num_parameters(self._input_dim+self._additional_dim, self._k_start))
        
        if self._additional_dim == 1:             
            C = C_param[:,np.newaxis,:]
            RK_1 = RK_1_param[:,np.newaxis,:]
            RK_2 = RK_2_param[:,np.newaxis,:]
        else: 
            C = self.get_rotation_tensor(self._input_dim, C_param)[:,-self._additional_dim:,:]
            RK_1 = self.get_rotation_tensor(self._additional_dim, RK_1_param)
            RK_2 = self.get_rotation_tensor(self._additional_dim, RK_2_param)
        
        RN = self.get_rotation_tensor(self._input_dim, RN_param)
        RG = self.get_rotation_tensor(self._input_dim+self._additional_dim, RG_param)

        if self._manifold_nonlinearity == tf.nn.tanh or self._manifold_nonlinearity == tf.nn.sigmoid or self._manifold_nonlinearity == tf.nn.relu :
            NonlinK_param = None
        elif manifold_nonlinearity == helper.parametric_relu:
            NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, 2*self._additional_dim)
        elif manifold_nonlinearity == helper.polinomial_nonlin:
            NonlinK_param, param_index = helper.slice_parameters(self._parameters, param_index, (self._polinomial_degree+1)*self._additional_dim)
        
        # C*z
        if C.get_shape()[0].value == 1: #one set of parameters
            Cz = tf.matmul(z0, C[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            Cz = tf.matmul(C, z0[:,:,np.newaxis])[:, :, 0]

        # RK1*C*z
        if RK_1.get_shape()[0].value == 1: #one set of parameters
            RK1Cz = tf.matmul(Cz, RK_1[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            RK1Cz = tf.matmul(RK_1, Cz[:,:,np.newaxis])[:, :, 0]

        pre_nonlinK = pre_bias+pre_scale*RK1Cz
        nonlinK, nonlinK_J = self.apply_manifold_nonlin(pre_nonlinK, NonlinK_param)
        post_nonlinK = post_bias+post_scale*nonlinK

        # RK2*nonlin(a(C*z)+b)
        if RK_2.get_shape()[0].value == 1: #one set of parameters
            y_k = tf.matmul(post_nonlinK, RK_2[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            y_k = tf.matmul(RK_2, post_nonlinK[:,:,np.newaxis])[:, :, 0]
        
        # RK2*nonlin(a(C*z)+b)
        if RN.get_shape()[0].value == 1: #one set of parameters
            y_n = tf.matmul(z0, RN[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            y_n = tf.matmul(RN, z0[:,:,np.newaxis])[:, :, 0]
        y = tf.concat([y_n, y_k], axis=-1)
        
        # full rotation
        if RK_2.get_shape()[0].value == 1: #one set of parameters
            z = tf.matmul(y, RG[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            z = tf.matmul(RG, y[:,:,np.newaxis])[:, :, 0]

        log_volume_increase_ratio = tf.reduce_sum(0.5*tf.log(1e-7+(nonlinK_J*pre_scale*post_scale)**2+1), axis=[1], keep_dims=True)
        log_pdf_z = log_pdf_z0 - log_volume_increase_ratio
        return z, log_pdf_z


class HouseholdRotationFlow():
    """
    Household Rotation Flow class.
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, k_start=1, init_reflection=1, name='household_rotation_transform'):   
        self._parameter_scale = 1.
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._k_start = k_start
        self._init_reflection = init_reflection

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim, k_start=1):  
        return sum(list(range(max(2, k_start), input_dim+1)))

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        self._parameters.get_shape().assert_is_compatible_with([None, HouseholdRotationFlow.required_num_parameters(self._input_dim, self._k_start)])

        batched_rot_matrix = helper.householder_rotations_tf(n=self.input_dim, batch=tf.shape(self._parameters)[0], k_start=self._k_start, init_reflection=self._init_reflection, params=self._parameters)        
        if batched_rot_matrix.get_shape()[0].value == 1: #one set of parameters
            z = tf.matmul(z0, batched_rot_matrix[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            z = tf.matmul(batched_rot_matrix, z0[:,:,np.newaxis])[:, :, 0]

        log_pdf_z = log_pdf_z0 
        return z, log_pdf_z

class LinearIARFlow():
    """
    Linear Inverse Autoregressive Flow class.

    This is only to be used with a non-centered diagonal-covariance gaussian to archive the full
    flexibility since it does the minimal, which is volume preserving skewing (and no scaling or translation).
    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    def __init__(self, parameters, input_dim, name='linearIAR_transform'):   
        self._parameter_scale = 1.
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._mask_mat_ones = helper.triangular_ones([self._input_dim, self._input_dim], trilmode=-1)
        self._diag_mat_ones = helper.block_diagonal_ones([self._input_dim, self._input_dim], [1, 1])

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    def required_num_parameters(input_dim):
        return input_dim*input_dim

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        self._parameters.get_shape().assert_is_compatible_with([None, LinearIARFlow.required_num_parameters(self._input_dim)])
        param_matrix = tf.reshape(self._parameters, [-1, self._input_dim, self._input_dim])
        mask_matrix = tf.reshape(self._mask_mat_ones, [1, self._input_dim, self._input_dim]) 
        diag_matrix = tf.reshape(self._diag_mat_ones, [1, self._input_dim, self._input_dim]) 
        cho = mask_matrix*param_matrix+diag_matrix
        if cho.get_shape()[0].value == 1: #one set of parameters
            z = tf.matmul(z0, cho[0, :, :], transpose_a=False, transpose_b=True)
        else: # batched parameters
            z = tf.matmul(cho, z0[:,:,np.newaxis])[:, :, 0]
        log_pdf_z = log_pdf_z0 
        return z, log_pdf_z

class NonLinearIARFlow():
    """
    Non-Linear Inverse Autoregressive Flow class.

    Args:
      parameters: parameters of transformation all appended.
      input_dim : input dimensionality of the transformation. 
    Raises:
      ValueError: 
    """
    # def __init__(self, parameters, input_dim, layer_expansions=[25,25], b_RNN_style=False,  name='nonlinearIAR_transform'):   #manifold
    # def __init__(self, parameters, input_dim, layer_expansions=[2,2], b_RNN_style=True,  name='nonlinearIAR_transform'):   #mnist
    # def __init__(self, parameters, input_dim, layer_expansions=[1,1], b_RNN_style=True,  name='nonlinearIAR_transform'):   #real 1
    def __init__(self, parameters, input_dim, layer_expansions=[1,], b_RNN_style=True, last_noise=False, name='nonlinearIAR_transform'):   #real
        # self._parameter_scale = 5
        self._parameter_scale = 1
        self._parameters = self._parameter_scale*parameters
        self._input_dim = input_dim
        self._layer_expansions = layer_expansions
        self._nonlinearity = helper.lrelu
        self._b_RNN_style = b_RNN_style
        self._last_noise = last_noise

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._input_dim

    @staticmethod
    # def required_num_parameters(input_dim, layer_expansions=[25,25]): #manifold
    # def required_num_parameters(input_dim, layer_expansions=[2,2]): #mnist
    # def required_num_parameters(input_dim, layer_expansions=[1,1]): # real 1
    def required_num_parameters(input_dim, layer_expansions=[1,]):
        n_parameters = 0
        concat_layer_expansions = [1, *layer_expansions, 2]
        for l in range(len(concat_layer_expansions)-1):
            n_parameters += concat_layer_expansions[l]*(input_dim-1)*concat_layer_expansions[l+1]*(input_dim-1) # matrix
            n_parameters += concat_layer_expansions[l+1]*(input_dim-1) # bias
        n_parameters += 2
        return n_parameters

    def MADE_forward(self, input_x, layerwise_parameters, masks, nonlinearity):
        input_dim = input_x.get_shape().as_list()[1]
        curr_input = input_x
        for l in range(len(layerwise_parameters)):
            W, bias = layerwise_parameters[l]
            W_masked = W
            # W_masked = W*masks[l]
            if W.get_shape()[0].value == 1: #one set of parameters
                affine = tf.matmul(curr_input, W_masked[0, :, :], transpose_a=False, transpose_b=True)
            else: # batched parameters
                affine = tf.matmul(W_masked, curr_input[:,:,np.newaxis])[:, :, 0]
            if l < len(layerwise_parameters)-1: curr_input = nonlinearity(affine+bias)
            else: curr_input = (affine+bias)

        mu = tf.slice(curr_input, [0, 0], [-1, input_dim])
        log_sig = tf.slice(curr_input, [0, input_dim], [-1, input_dim])
        return mu, log_sig

    def transform(self, z0, log_pdf_z0):
        verify_size(z0, log_pdf_z0)
        self._parameters.get_shape().assert_is_compatible_with([None, NonLinearIARFlow.required_num_parameters(self._input_dim, self._layer_expansions)])
        mask_tensors = helper.tf_get_mask_list_for_MADE(self._input_dim-1, self._layer_expansions, add_mu_log_sigma_layer=True)
        z0_1 = tf.slice(z0, [0, 0], [-1, 1])
        z0_2toD = tf.slice(z0, [0, 1], [-1, self._input_dim-1])
        z0_1toDminus1 = tf.slice(z0, [0, 0], [-1, self._input_dim-1])

        mu_for_1 = tf.slice(self._parameters, [0, 0], [-1, 1])
        log_sig_for_1 = tf.slice(self._parameters, [0, 1], [-1, 1])
        start_ind = 2
        concat_layer_expansions = [1, *self._layer_expansions, 2]
        layerwise_parameters = []
        for l in range(len(concat_layer_expansions)-1):
            W_num_param = concat_layer_expansions[l]*(self._input_dim-1)*concat_layer_expansions[l+1]*(self._input_dim-1) # matrix
            B_num_param = concat_layer_expansions[l+1]*(self._input_dim-1) # bias
            W_l_flat = tf.slice(self._parameters, [0, start_ind], [-1, W_num_param])
            B_l_flat = tf.slice(self._parameters, [0, start_ind+W_num_param], [-1, B_num_param])
            W_l = tf.reshape(W_l_flat, [-1, concat_layer_expansions[l+1]*(self._input_dim-1), concat_layer_expansions[l]*(self._input_dim-1)])              
            B_l = tf.reshape(B_l_flat, [-1, concat_layer_expansions[l+1]*(self._input_dim-1)])
            layerwise_parameters.append((W_l, B_l))  
            start_ind += (W_num_param+B_num_param)
        
        mu_for_2toD, log_sig_for_2toD = self.MADE_forward(z0_1toDminus1, layerwise_parameters, mask_tensors, self._nonlinearity)
        if self._b_RNN_style:
            m_for_2toD, s_for_2toD, m_for_1, s_for_1 = mu_for_2toD, log_sig_for_2toD, mu_for_1, log_sig_for_1
            
            if self._last_noise: sigm_for_1, sigm_for_2toD = 0.01*tf.nn.sigmoid(s_for_1), 0.01*tf.nn.sigmoid(s_for_2toD)
            else: sigm_for_1, sigm_for_2toD = tf.nn.sigmoid(s_for_1), tf.nn.sigmoid(s_for_2toD)

            z_2toD = sigm_for_2toD*z0_2toD+(1-sigm_for_2toD)*m_for_2toD
            z_1 = sigm_for_1*z0_1+(1-sigm_for_1)*m_for_1 
            z = tf.concat([z_1, z_2toD], axis=1)
            log_abs_det_jacobian = tf.log(1e-7+sigm_for_1)+tf.reduce_sum(tf.log(1e-7+sigm_for_2toD), axis=[1], keep_dims=True)
        else:
            log_scale_for_2toD = log_sig_for_2toD-1
            log_scale_for_1 = log_sig_for_1-1
            
            scale_for_2toD = tf.exp(log_scale_for_2toD)
            scale_for_1 = tf.exp(log_scale_for_1)

            z_2toD = mu_for_2toD+scale_for_2toD*z0_2toD
            z_1= mu_for_1+scale_for_1*z0_1
            z = tf.concat([z_1, z_2toD], axis=1)
            log_abs_det_jacobian = tf.log(1e-7+scale_for_1)+tf.reduce_sum(tf.log(1e-7+scale_for_2toD), axis=[1], keep_dims=True)
            # log_abs_det_jacobian = log_scale_for_1+tf.reduce_sum(log_scale_for_2toD, axis=[1], keep_dims=True)
        # else:
        #     z_2toD = mu_for_2toD+tf.nn.sigmoid(log_sig_for_2toD)*z0_2toD
        #     z_1= mu_for_1+tf.nn.sigmoid(log_sig_for_1)*z0_1
        #     z = tf.concat([z_1, z_2toD], axis=1)
        #     log_abs_det_jacobian = tf.log(tf.nn.sigmoid(log_sig_for_1))+tf.reduce_sum(tf.log(tf.nn.sigmoid(log_sig_for_2toD)), axis=[1], keep_dims=True)
        # else:
        #     z_2toD = mu_for_2toD+tf.nn.softplus(log_sig_for_2toD)*z0_2toD
        #     z_1= mu_for_1+tf.nn.softplus(log_sig_for_1)*z0_1
        #     z = tf.concat([z_1, z_2toD], axis=1)
        #     log_abs_det_jacobian = tf.log(1e-7+tf.nn.softplus(log_sig_for_1))+tf.reduce_sum(tf.log(1e-7+tf.nn.softplus(log_sig_for_2toD)), axis=[1], keep_dims=True)


        log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
        return z, log_pdf_z
        # return m_for_2toD, sigm_for_2toD

def verify_size(z0, log_pdf_z0):
  z0.get_shape().assert_is_compatible_with([None, None])
  if log_pdf_z0 is not None:
    log_pdf_z0.get_shape().assert_is_compatible_with([None, 1])

def _jacobian(y, x):
    batch_size = y.get_shape()[0].value
    flat_y = tf.reshape(y, [batch_size, -1])
    num_y = flat_y.get_shape()[1].value
    one_hot = np.zeros((batch_size, num_y))
    jacobian_rows = []
    for i in range(num_y):
        one_hot.fill(0)
        one_hot[:, i] = 1
        grad_flat_y = tf.constant(one_hot, dtype=y.dtype)

        grad_x, = tf.gradients(flat_y, [x], grad_flat_y, gate_gradients=True)
        assert grad_x is not None, "Variable `y` is not computed from `x`."

        row = tf.reshape(grad_x, [batch_size, 1, -1])
        jacobian_rows.append(row)

    return tf.concat(jacobian_rows, 1)

def _log_determinant(matrices):
    _, logdet = np.linalg.slogdet(matrices)
    return logdet.reshape(len(matrices), 1)

def _check_logdet(flow, z0, log_pdf_z0, rtol=1e-5):
    z1, log_pdf_z1 = flow.transform(z0, log_pdf_z0)
    jacobian = _jacobian(z1, z0)
    
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()  
    sess.run(init)
    out_jacobian, out_log_pdf_z0, out_log_pdf_z1 = sess.run([jacobian, log_pdf_z0, log_pdf_z1])

    # The logdet will be ln|det(dz1/dz0)|.
    logdet_expected = _log_determinant(out_jacobian)
    logdet = out_log_pdf_z0 - out_log_pdf_z1

    # if np.allclose(logdet_expected, logdet, rtol=rtol):
    if np.all(np.abs(logdet_expected-logdet)<rtol):
        print('Transform update correct.')
    else: 
        print('Transform update incorrect!!!!!!!!!!!!!!!!')
        # np.abs(logdet_expected-logdet) 1e-08+rtol*np.abs(logdet)
        pdb.set_trace()

# batch_size = 5
# n_latent = 400
# name = 'transform'
# transform_to_check = NonLinearIARFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
# try: transform_object =  transform_to_check(parameters, n_latent)
# except: transform_object =  transform_to_check(None, n_latent)
# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32)

# nonlinearIAF_transform1 = NonLinearIARFlow(parameters, n_latent)
# m_for_2toD, sigm_for_2toD = nonlinearIAF_transform1.transform(z0, log_pdf_z0)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# m_for_2toD, sigm_for_2toD = sess.run([m_for_2toD, sigm_for_2toD])
# pdb.set_trace()


# batch_size = 5
# n_latent = 400
# name = 'transform'
# transform_to_check = NonLinearIARFlow
# n_parameter = transform_to_check.required_num_parameters(n_latent)

# parameters = 10*tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
# try: transform_object =  transform_to_check(parameters, n_latent)
# except: transform_object =  transform_to_check(None, n_latent)
# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32)
# for repeat in range(5): _check_logdet(transform_object, z0, log_pdf_z0)

# parameters = 1*tf.layers.dense(inputs = tf.ones(shape=(batch_size, 1)), units = n_parameter, use_bias = False, activation = None)
# try: transform_object =  transform_to_check(parameters, n_latent)
# except: transform_object =  transform_to_check(None, n_latent)
# z0 = tf.random_normal((batch_size, n_latent), 0, 1, dtype=tf.float32)
# log_pdf_z0 = tf.random_normal((batch_size, 1), 0, 1, dtype=tf.float32)
# for repeat in range(5): _check_logdet(transform_object, z0, log_pdf_z0)

# pdb.set_trace()

# dim = 3
# batch = 1000 
# k_start = 1

# params = tf.random_normal((batch, sum(list(range(max(2, k_start), dim+1)))), 0, 1, dtype=tf.float32)
# # params = tf.ones((batch, sum(list(range(max(2, k_start), dim+1)))))
# # params = None

# z0 = tf.tile(np.asarray([[1., 1., 1.]]).astype(np.float32), [batch, 1])
# log_pdf_z0 = tf.random_normal((batch, 1), 0, 1, dtype=tf.float32)

# rotation_transform1 = HouseholdRotationFlow(params, dim, k_start=k_start, init_reflection=1)
# rotation_transform2 = HouseholdRotationFlow(params, dim, k_start=k_start, init_reflection=-1)
# z1, log_pdf_z1 = rotation_transform1.transform(z0, log_pdf_z0)
# z2, log_pdf_z2 = rotation_transform2.transform(z0, log_pdf_z0)

# init = tf.initialize_all_variables()
# sess = tf.InteractiveSession()  
# sess.run(init)
# z1_np, log_pdf_z1_np, z2_np, log_pdf_z2_np = sess.run([z1, log_pdf_z1, z2, log_pdf_z2])
# helper.dataset_plotter([z1_np, z2_np], show_also=True)
# pdb.set_trace()



# # self.prior_map = f_p(n_latent | n_state, n_context). f_p(z_t | h<t, e(c_t))
# class TransformMap():
#     def __init__(self, config, name = '/TransformMap'):
#         self.name = name
#         self.config = config
#         self.constructed = False
 
#     def forward(self, transform_class_list, name = ''):
#         with tf.variable_scope("TransformMap", reuse=self.constructed):
#             input_dim = self.config['n_latent']
#             transforms_list = []
#             for transform_to_use in transform_class_list:  
#                 n_parameter = transform_to_use.required_num_parameters(input_dim)
#                 if n_parameter>0:
#                     parameters = tf.layers.dense(inputs = tf.ones(shape=(1, 1)), units = n_parameter, use_bias = False, activation = None)
#                 else: parameters = None
#                 transforms_list.append(transform_to_use(parameters, input_dim))
#             self.constructed = True
#             return transforms_list



# class ProjectiveFlow():
#     """
#     Projective Flow class.
#     Args:
#       parameters: parameters of transformation all appended.
#       input_dim : input dimensionality of the transformation. 
#     Raises:
#       ValueError: 
#     """
#     def __init__(self, parameters, input_dim, additional_dim=3, k_start=1, init_reflection=1, name='projective_transform'):   
#         self._parameter_scale = 1.
#         self._parameters = self._parameter_scale*parameters
#         self._input_dim = input_dim
#         self._additional_dim = additional_dim
#         self._k_start = k_start
#         self._init_reflection = init_reflection
#         self._manifold_nonlinearity = tf.nn.tanh

#     @property
#     def input_dim(self):
#         return self._input_dim

#     @property
#     def output_dim(self):
#         return self._input_dim+self._additional_dim

#     @staticmethod
#     def required_num_parameters(input_dim, additional_dim=3, k_start=1):  
#         if input_dim>=additional_dim: #row independent
#             if additional_dim == 1: n_basis_param = input_dim 
#             else: n_basis_param = sum(list(range(max(2, k_start), input_dim+1)))
#             return additional_dim+n_basis_param
#         else: #additional_dim > input_dim, column independent
#             if input_dim == 1: n_basis_param = additional_dim 
#             else: n_basis_param = sum(list(range(max(2, k_start), additional_dim+1)))
#             return input_dim+n_basis_param

#     def transform(self, z0, log_pdf_z0):
#         verify_size(z0, log_pdf_z0)
#         self._parameters.get_shape().assert_is_compatible_with([None, ProjectiveFlow.required_num_parameters(self._input_dim, self._additional_dim, self._k_start)])

#         if self._input_dim>=self._additional_dim: 
#             shear_param = tf.slice(self._parameters, [0, 0], [-1, self._additional_dim])
#             basis_param = tf.slice(self._parameters, [0, self._additional_dim], [-1, -1])
#         else: 
#             shear_param = tf.slice(self._parameters, [0, 0], [-1, self._input_dim])
#             basis_param = tf.slice(self._parameters, [0, self._input_dim], [-1, -1])

#         shear_matrix = tf.nn.softplus(shear_param)
#         if self._input_dim>=self._additional_dim: 
#             if self._additional_dim == 1: basis_tensor = (basis_param/tf.sqrt(tf.reduce_sum(basis_param**2, axis=[-1], keep_dims=True)))[:,np.newaxis,:]
#             else: basis_tensor = helper.householder_rotations_tf(n=self.input_dim, batch=tf.shape(basis_param)[0], k_start=self._k_start, 
#                                                                  init_reflection=self._init_reflection, params=basis_param)[:,-self._additional_dim:,:]
#         else: 
#             if self._input_dim == 1: basis_tensor = (basis_param/tf.sqrt(tf.reduce_sum(basis_param**2, axis=[-1], keep_dims=True)))[:,:,np.newaxis]
#             else: basis_tensor = helper.householder_rotations_tf(n=self._additional_dim, batch=tf.shape(basis_param)[0], k_start=self._k_start, 
#                                                                  init_reflection=self._init_reflection, params=basis_param)[:,:,-self.input_dim:]
#         # Transformation
#         if self._input_dim>=self._additional_dim: z_project_input = z0
#         else: z_project_input = shear_matrix*z0

#         if basis_tensor.get_shape()[0].value == 1: #one set of parameters
#             z_project = tf.matmul(z_project_input, basis_tensor[0, :, :], transpose_a=False, transpose_b=True)
#         else: # batched parameters
#             z_project = tf.matmul(basis_tensor, z_project_input[:,:,np.newaxis])[:, :, 0]
        
#         if self._input_dim>=self._additional_dim: z_project_sheared = tf.concat([z0, shear_matrix*z_project], axis=1)  
#         else: z_project_sheared = tf.concat([z0, z_project], axis=1)           
#         z = self._manifold_nonlinearity(z_project_sheared)

#         # Density Update
#         if self._manifold_nonlinearity == tf.nn.tanh:
#             diagonal_nonlinearity_jacobian = 1-z**2
#         else: pdb.set_trace()

#         diagonal_nonlinearity_jacobian_1 = tf.slice(diagonal_nonlinearity_jacobian, [0, 0], [-1, self._input_dim])
#         diagonal_nonlinearity_jacobian_2 = tf.slice(diagonal_nonlinearity_jacobian, [0, self._input_dim], [-1, -1])
#         pdb.set_trace()

        
#         log_abs_det_jacobian = 0
#         log_pdf_z = log_pdf_z0 - log_abs_det_jacobian
#         return z, log_pdf_z

