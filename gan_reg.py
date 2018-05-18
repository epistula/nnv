import sys

import inspect
import traceback

import gc
import os
import pdb
import time
import shutil
import argparse
import helper
from sklearn.manifold import TSNE
from umap import UMAP
import tensorflow as tf


# dataset_to_use = 'IMAGENET'
# dataset_to_use = 'BEDROOM'
# dataset_to_use = 'CELEB'
# dataset_to_use = 'CIFAR10'
dataset_to_use = 'MNIST'
# dataset_to_use = 'CAT'
# dataset_to_use = 'FLOWERS'
# dataset_to_use = 'CUB'
# dataset_to_use = 'TOY'
# dataset_to_use = 'INTENSITY'

Algorithm = 'WAEVanilla'
if Algorithm == 'WAE':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9,  
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 1, 'n_filter': 128, 
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'NS-GAN', 'dual_dist_mode': '', 
                             'enc_normalization_mode': 'Layer Norm', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'Layer Norm', 
                             'enc_reg_strength': 10, 'enc_inv_MMD_n_reflect': 3, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 10, 
                             'critic_reg_mode': [], 'cri_reg_strength': 0, 'lambda_mix': 0.25}
elif Algorithm == 'WAEVanilla':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0., 'beta2': 0.9,  
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 1, 'n_filter': 32, 
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'INV-MMD2', 'dual_dist_mode': '', 
                             'enc_normalization_mode': 'Layer Norm', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'None', 
                             # 'enc_normalization_mode': 'None', 'gen_normalization_mode': 'None', 'cri_normalization_mode': 'None', 
                             'enc_reg_strength': 5, 'enc_inv_MMD_n_reflect': 1, 'enc_inv_MMD_n_trans': 20, 'enc_inv_MMD_strength': 100, 
                             'critic_reg_mode': [], 'cri_reg_strength': 0, 'lambda_mix': 0.25}
elif Algorithm == 'WGANGP':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9,
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 5, 'n_filter': 128,
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'NS-GAN', 'dual_dist_mode': 'Prior', 
                             'enc_normalization_mode': 'Layer Norm', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'Layer Norm', 
                             'enc_reg_strength': 20, 'enc_inv_MMD_n_reflect': 3, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 10, 
                             'critic_reg_mode': ['Trivial Gradient Norm',], 'cri_reg_strength': 10, 'lambda_mix': 0.25}
elif Algorithm == 'PDWGAN':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9,
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 5, 'n_filter': 128,
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'NS-GAN', 'dual_dist_mode': 'Prior', 
                             'enc_normalization_mode': 'Layer Norm', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'Layer Norm', 
                             'enc_reg_strength': 20, 'enc_inv_MMD_n_reflect': 3, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 10, 
                             'critic_reg_mode': ['Coupling Gradient Vector',], 'cri_reg_strength': 0.1, 'lambda_mix': 0.25}
elif Algorithm == 'WGANGPCannon':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9,
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 5, 'n_filter': 128,
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'MMD', 'dual_dist_mode': 'Prior', 
                             'enc_normalization_mode': 'None', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'None', 
                             'enc_reg_strength': 100, 'enc_inv_MMD_n_reflect': 3, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 10, 
                             'critic_reg_mode': ['Trivial Gradient Norm',], 'cri_reg_strength': 10, 'lambda_mix': 0.25}
elif Algorithm == 'PDWGANCannon':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9,
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 5, 'n_filter': 128,
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'MMD', 'dual_dist_mode': 'Prior', 
                             'enc_normalization_mode': 'None', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'None', 
                             'enc_reg_strength': 100, 'enc_inv_MMD_n_reflect': 3, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 10, 
                             'critic_reg_mode': ['Coupling Gradient Vector',], 'cri_reg_strength': 1, 'lambda_mix': 0.25}
elif Algorithm == 'PDWGANCannon2':
    alg_specific_settings = {'optimizer_class': 'Adam', 'learning_rate': 1e-4, 'beta1': 0.5, 'beta2': 0.9,
                             'rel_enc_skip_rate': 1, 'rel_cri_skip_rate': 1, 'rel_gen_skip_rate': 5, 'n_filter': 128,
                             'encoder_mode': 'UnivApprox', 'divergence_mode': 'MMD', 'dual_dist_mode': 'CouplingAndPrior', 
                             'enc_normalization_mode': 'None', 'gen_normalization_mode': 'Batch Norm', 'cri_normalization_mode': 'None', 
                             'enc_reg_strength': 10, 'enc_inv_MMD_n_reflect': 3, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 10, 
                             'critic_reg_mode': ['Coupling Gradient Vector',], 'cri_reg_strength': 1, 'lambda_mix': 0.5}


global_experiment_name = 'EEEexperimentsStable-'+Algorithm+'-'

parser = argparse.ArgumentParser(description='Tensorflow Gan Models')
parser.add_argument('--exp_dir_postfix', type=str, default='', help='Directory to put the experiment postfix.')
parser.add_argument('--save_checkpoints', type=bool, default=False, help='store the checkpoints?')
parser.add_argument('--restore_dir', type=str, default='/15fc2c543c5c4d958a733f646d8e8abe/checkpoint/', help='Directory of restore experiment.')
parser.add_argument('--restore', type=bool, default=False, help='Restore model.')
parser.add_argument('--gpu', type=str, default='0', help='gpu to use.')
parser.add_argument('--epochs', type=int, default=1000000000, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=50, help='Input batch size for training.')
parser.add_argument('--time_steps', type=int, default=1, help='Number of timesteps')
parser.add_argument('--seed', type=int, default=1123124, help='random seed')
    
if dataset_to_use == 'IMAGENET':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'IMAGENET', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=300, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[100,10], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[20,1], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[20,1], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,0], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[1,0], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[1,0], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[0,1], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')
    

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.ImagenetLoader import DataLoader
    # # # # # #############################################################################################################################

if dataset_to_use == 'BEDROOM':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'BEDROOM', help='Directory to put the experiments.')
    
    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=300, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[100,10], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[20,1], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[20,1], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,0], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[1,0], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[1,0], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[1,0], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[1,0], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=100, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.BedroomLoader import DataLoader
    # # # # # #############################################################################################################################

elif dataset_to_use == 'CELEB':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'CELEB', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=300, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[100,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[200,0], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[5,4], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,0], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[1,0], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[1,0], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[20,1], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=64, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.CelebALoader import DataLoader
    # # # # # #############################################################################################################################

elif dataset_to_use == 'CAT':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'CAT', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=40, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=50, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[500,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[500,0], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[50,10], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[3,1], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[3,1], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[3,1], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[100,10], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=32, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.CatLoader import DataLoader
    # # # # # #############################################################################################################################

elif dataset_to_use == 'FLOWERS':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'FLOWERS', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=40, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=50, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[500,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[500,0], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[50,10], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[3,1], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[3,1], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[3,1], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[100,10], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=32, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')
    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.FlowersLoader import DataLoader
    # # # # #############################################################################################################################

elif dataset_to_use == 'CUB':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'CUB', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=40, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=50, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[500,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[500,0], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[50,10], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[3,1], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[3,1], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[3,1], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[100,10], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=32, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.CubLoader import DataLoader
    # # #############################################################################################################################

elif dataset_to_use == 'CIFAR10':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'CIFAR10', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=200, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[300,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[300,0], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[25,10], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[3,1], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[3,1], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[3,1], help='fixed samples epoch repeat')
    # parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[50,10], help='compute fid and inception score')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[0,1], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    # parser.add_argument('--n_latent', type=int, default=48, help='n_latent.')
    parser.add_argument('--n_latent', type=int, default=64, help='n_latent.')
    # parser.add_argument('--n_latent', type=int, default=alg_specific_settings['n_latent'], help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.Cifar10Loader import DataLoader
    #############################################################################################################################

elif dataset_to_use == 'MNIST':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'MNIST', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=200, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[300,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[300,0], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[25,10], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[3,1], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[3,1], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[3,1], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[0,1], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    # parser.add_argument('--n_latent', type=int, default=16, help='n_latent.')
    parser.add_argument('--n_latent', type=int, default=64, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.ColorMnistLoader import DataLoader
    #############################################################################################################################

elif dataset_to_use == 'INTENSITY':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'INTENSITY', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=15, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=1, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[3000,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[30,5], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[30,5], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[30,5], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[30,5], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[30,5], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[0,1], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=2, help='n_latent.')
    # parser.add_argument('--n_latent', type=int, default=64, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=200, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.IntensityToyDataLoader import DataLoader
    #############################################################################################################################

elif dataset_to_use == 'TOY':
    parser.add_argument('--global_exp_dir', type=str, default='./'+global_experiment_name+'TOY', help='Directory to put the experiments.')

    parser.add_argument('--gradient_clipping', type=float, default=0, help='Initial weight decay.')
    parser.add_argument('--optimizer_class', type=str, default=alg_specific_settings['optimizer_class'], help='Optimizer type.')
    parser.add_argument('--learning_rate', type=float, default=alg_specific_settings['learning_rate'], help='Initial learning rate.')
    parser.add_argument('--beta1', type=float, default=alg_specific_settings['beta1'], help='Momentum Beta 1.')
    parser.add_argument('--beta2', type=float, default=alg_specific_settings['beta2'], help='Momentum Beta 2.')
    parser.add_argument('--rel_enc_skip_rate', type=float, default=alg_specific_settings['rel_enc_skip_rate'], help='Relative encoder skip steps.')
    parser.add_argument('--rel_cri_skip_rate', type=float, default=alg_specific_settings['rel_cri_skip_rate'], help='Relative critic skip steps.')
    parser.add_argument('--rel_gen_skip_rate', type=float, default=alg_specific_settings['rel_gen_skip_rate'], help='Relative generator skip steps.')

    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--vis_interval', type=int, default=20, help='how many batches to wait before visualizing training status')
    parser.add_argument('--in_between_vis', type=int, default=0, help='how many reports to wait before visualizing training status')
    parser.add_argument('--test_epoch_rate', type=list, default=[20,1], help='test epoch repeat')
    parser.add_argument('--latent_vis_TSNE_epoch_rate', type=list, default=[1,-1], help='latent epoch repeat')
    parser.add_argument('--latent_vis_UMAP_epoch_rate', type=list, default=[1,-1], help='latent epoch repeat')
    parser.add_argument('--reconst_vis_epoch_rate', type=list, default=[1,0], help='reconst epoch repeat')
    parser.add_argument('--interpolate_vis_epoch_rate', type=list, default=[1,0], help='interpolation epoch repeat')
    parser.add_argument('--fixed_samples_vis_epoch_rate', type=list, default=[1,0], help='fixed samples epoch repeat')
    parser.add_argument('--fid_inception_score_epoch_rate', type=list, default=[0,1], help='compute fid and inception score')
    parser.add_argument('--pigeonhole_score_epoch_rate', type=list, default=[0,1], help='compute pigeonhole score')

    parser.add_argument('--n_context', type=int, default=1, help='n_context.')
    parser.add_argument('--n_state', type=int, default=1, help='n_state.')
    parser.add_argument('--n_latent', type=int, default=50, help='n_latent.')
    parser.add_argument('--n_filter', type=int, default=alg_specific_settings['n_filter'], help='n_filter.')
    parser.add_argument('--n_flat', type=int, default=400, help='n_flat.')
    
    parser.add_argument('--div_activation_function', type=object, default=helper.lrelu, help='activation function for Diverger.')
    parser.add_argument('--enc_activation_function', type=object, default=helper.lrelu, help='activation function for Encoder.')
    parser.add_argument('--gen_activation_function', type=object, default=helper.lrelu, help='activation function for Generator.')
    parser.add_argument('--cri_activation_function', type=object, default=helper.lrelu, help='activation function for Critic.')
    parser.add_argument('--div_normalization_mode', type=str, default='None', help='normalization mode for Diverger.')
    parser.add_argument('--enc_normalization_mode', type=str, default=alg_specific_settings['enc_normalization_mode'], help='normalization mode for Encoder.')
    parser.add_argument('--gen_normalization_mode', type=str, default=alg_specific_settings['gen_normalization_mode'], help='normalization mode for Generator.')
    parser.add_argument('--cri_normalization_mode', type=str, default=alg_specific_settings['cri_normalization_mode'], help='normalization mode for Critic.')
    
    parser.add_argument('--sample_distance_mode', type=str, default='Euclidean', help='sample distance mode.')
    parser.add_argument('--kernel_mode', type=str, default='InvMultiquadratics', help='kernel mode.')
    parser.add_argument('--encoder_mode', type=str, default=alg_specific_settings['encoder_mode'], help='encoder mode.')
    parser.add_argument('--divergence_mode', type=str, default=alg_specific_settings['divergence_mode'], help='divergence mode.')
    parser.add_argument('--dual_dist_mode', type=str, default=alg_specific_settings['dual_dist_mode'], help='dual distribution mode.')
    parser.add_argument('--lambda_mix', type=str, default=alg_specific_settings['lambda_mix'], help='mixture amount for coupling.')
    parser.add_argument('--critic_reg_mode', type=list, default=alg_specific_settings['critic_reg_mode'], help='critic regularizer mode.')
    parser.add_argument('--enc_reg_strength', type=float, default=alg_specific_settings['enc_reg_strength'], help='encoder regularization strength')
    parser.add_argument('--enc_inv_MMD_n_reflect', type=float, default=alg_specific_settings['enc_inv_MMD_n_reflect'], help='encoder invariant MMD num of reflections')
    parser.add_argument('--enc_inv_MMD_n_trans', type=float, default=alg_specific_settings['enc_inv_MMD_n_trans'], help='encoder invariant MMD num of transforms')
    parser.add_argument('--enc_inv_MMD_strength', type=float, default=alg_specific_settings['enc_inv_MMD_strength'], help='encoder invariant MMD strength')
    parser.add_argument('--cri_reg_strength', type=float, default=alg_specific_settings['cri_reg_strength'], help='cririminator regularization strength')
    parser.add_argument('--enc_sine_freq', type=float, default=2, help='encoder sine frequency')

    global_args = parser.parse_args()
    global_args.curr_epoch = 1
    from datasetLoaders.ToyDataLoader import DataLoader
    # from datasetLoaders.RandomManifoldDataLoader import DataLoader
    ##############################################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = global_args.gpu
print("os.environ['CUDA_VISIBLE_DEVICES'], ", os.environ['CUDA_VISIBLE_DEVICES'])

if Algorithm == 'WAE' or Algorithm == 'WAEVanilla': 
    from models.WAE.Model import Model
elif Algorithm == 'WGANGP' or Algorithm == 'PDWGAN':
    from models.PDWGAN.Model import Model
elif Algorithm == 'WGANGPCannon' or Algorithm == 'PDWGANCannon' or Algorithm == 'PDWGANCannon2':
    from models.PDWGANCannon.Model import Model
elif Algorithm == 'PDWGANTest':
    from models.PDWGAN.ModelTest import Model
from models.EVAL.InceptionScore import InceptionScore

import distributions 
import helper
import random
import numpy as np
import tensorflow as tf
random.seed(global_args.seed)
np.random.seed(global_args.seed)
tf.set_random_seed(global_args.seed)

data_loader = DataLoader(batch_size = global_args.batch_size, time_steps = global_args.time_steps)
global_args.exp_dir = helper.get_exp_dir(global_args)
helper.list_hyperparameters(global_args.exp_dir)

print("TENSORBOARD: Linux:\npython -m tensorflow.tensorboard --logdir=model1:"+\
    os.path.realpath(global_args.exp_dir)+" --port="+str(20000+int(global_args.exp_dir[-4:-1], 16))+" &")
print("TENSORBOARD: Mac:\nhttp://0.0.0.0:"+str(20000+int(global_args.exp_dir[-4:-1], 16)))
print("\n\n\n")

_, _, batch = next(data_loader)
try: 
    fixed_batch_data = batch['observed']['data']['image'].copy()
    random_batch_data = batch['observed']['data']['image'].copy()
except: pass
with tf.Graph().as_default():
    tf.set_random_seed(global_args.seed)
    model = Model(vars(global_args))

    global_step = tf.Variable(0.0, name='global_step', trainable=False)
    with tf.variable_scope("training"):
        tf.set_random_seed(global_args.seed)
        
        additional_inputs_tf = tf.placeholder(tf.float32, [2])
        batch_tf, input_dict_func = helper.tf_batch_and_input_dict(batch, additional_inputs_tf)
        model.inference(batch_tf, additional_inputs_tf)
        model.generative_model(batch_tf, additional_inputs_tf)

        div_vars = [v for v in tf.trainable_variables() if 'Diverger' in v.name]
        enc_vars = [v for v in tf.trainable_variables() if 'Encoder' in v.name] 
        gen_vars = [v for v in tf.trainable_variables() if 'Generator' in v.name] 
        cri_vars = [v for v in tf.trainable_variables() if 'Critic' in v.name]

        # Weight clipping
        if len(cri_vars)>0:
            cri_vars_flat_concat = tf.concat([tf.reshape(e, [-1]) for e in cri_vars], axis=0)
            max_abs_cri_vars = tf.reduce_max(tf.abs(cri_vars_flat_concat))
            clip_op_list = []
            for e in cri_vars: clip_op_list.append(tf.assign(e, tf.clip_by_value(e, -0.01, 0.01)))

    div_step_tf, enc_step_tf, cri_step_tf, gen_step_tf = None, None, None, None
    if global_args.optimizer_class == 'Adam':
        if hasattr(model, 'div_cost'):
            div_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                          learning_rate=global_args.learning_rate, beta1=global_args.beta1, beta2=global_args.beta2, epsilon=1e-08), 
                          loss=model.div_cost, var_list=div_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
        if hasattr(model, 'enc_cost'):
            enc_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                          learning_rate=global_args.learning_rate, beta1=global_args.beta1, beta2=global_args.beta2, epsilon=1e-08), 
                          loss=model.enc_cost, var_list=enc_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
        if hasattr(model, 'cri_cost'):
            cri_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                           learning_rate=global_args.learning_rate, beta1=global_args.beta1, beta2=global_args.beta2, epsilon=1e-08), 
                           loss=model.cri_cost, var_list=cri_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
        if hasattr(model, 'gen_cost'):
            gen_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.AdamOptimizer(
                          learning_rate=global_args.learning_rate, beta1=global_args.beta1, beta2=global_args.beta2, epsilon=1e-08), 
                          loss=model.gen_cost, var_list=gen_vars, global_step=global_step, clip_param=global_args.gradient_clipping)

    if not hasattr(model, 'div_cost'): model.div_cost = tf.zeros(shape=[1,])[0]
    if not hasattr(model, 'enc_cost'): model.enc_cost = tf.zeros(shape=[1,])[0]
    if not hasattr(model, 'cri_cost'): model.cri_cost = tf.zeros(shape=[1,])[0]
    if not hasattr(model, 'gen_cost'): model.gen_cost = tf.zeros(shape=[1,])[0]

    helper.variable_summaries(model.div_cost, '/div_cost')
    helper.variable_summaries(model.enc_cost, '/enc_cost')
    helper.variable_summaries(model.cri_cost, '/cri_cost')
    helper.variable_summaries(model.gen_cost, '/gen_cost')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    merged_summaries = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(global_args.exp_dir+'/summaries', sess.graph)
    sess.run(init)

    try:
        real_data_large_batch = None
        while real_data_large_batch is None or real_data_large_batch.shape[0]<400:
            _, _, batch = next(data_loader)
            if real_data_large_batch is None: real_data_large_batch = batch['observed']['data']['image'].copy()
            else: real_data_large_batch = np.concatenate([real_data_large_batch, batch['observed']['data']['image']], axis=0)
        helper.visualize_images2(real_data_large_batch[:int(np.sqrt(real_data_large_batch.shape[0]))**2, ...],
        block_size=[int(np.sqrt(real_data_large_batch.shape[0])), int(np.sqrt(real_data_large_batch.shape[0]))],
        save_dir=global_args.exp_dir+'Visualization/', postfix='_real_sample_only'+'_m')
    except: pass

    if global_args.restore:
        print("=> Loading checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
        try: 
            helper.load_checkpoint(saver, sess, global_args.global_exp_dir+global_args.restore_dir)  
            print("=> Loaded checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
        except: print("=> FAILED to load checkpoint: '{}'".format(global_args.global_exp_dir+global_args.restore_dir))
    
    if global_args.fid_inception_score_epoch_rate[0]>0:
        print('Loading inception model.')
        InceptionScoreModel = InceptionScore(sess)
        print('Successfully loaded inception model.')

    def get_scheduler():
        div_rate = 1
        if div_step_tf is not None: enc_rate = div_rate*global_args.rel_enc_skip_rate
        else: enc_rate = div_rate

        if enc_step_tf is not None: cri_rate = enc_rate*global_args.rel_cri_skip_rate
        else: cri_rate = enc_rate

        if cri_step_tf is not None: gen_rate = cri_rate*global_args.rel_gen_skip_rate
        else: gen_rate = cri_rate
        
        def scheduler(epoch, t):
            div_bool, enc_bool, cri_bool, gen_bool = False, False, False, False
            if t<3 or t % div_rate == 0: div_bool = True
            if t<3 or t % enc_rate == 0: enc_bool = True
            if t<3 or t % cri_rate == 0: cri_bool = True
            if t<3 or t % gen_rate == 0: gen_bool = True
            
            div_bool = div_bool and div_step_tf is not None
            enc_bool = enc_bool and enc_step_tf is not None
            cri_bool = cri_bool and cri_step_tf is not None
            gen_bool = gen_bool and gen_step_tf is not None

            return div_bool, enc_bool, cri_bool, gen_bool

        print('Div rate: ', div_rate, 'Enc rate: ', enc_rate, 'Cri rate: ', cri_rate, 'Gen rate: ', gen_rate)
        return scheduler

    scheduler = get_scheduler()
    all_step_tf = np.asarray([div_step_tf, enc_step_tf, cri_step_tf, gen_step_tf])
    all_cost_tf = np.asarray([model.div_cost, model.enc_cost, model.cri_cost, model.gen_cost])
    all_opt_keys = np.asarray(['div', 'enc', 'cri', 'gen'])
    optimize_bool = None

    def train():
        global optimize_bool, all_step_tf, all_cost_tf, all_opt_keys, random_batch_data, fixed_batch_data
        step_counts = {'div': 0, 'enc': 0, 'cri': 0, 'gen': 0}
        np_costs = {'div': 0, 'enc': 0, 'cri': 0, 'gen': 0}
        report_count = 0
        data_loader.train()
        train_div_loss_accum, train_enc_loss_accum, train_cri_loss_accum, train_gen_loss_accum, train_batch_size_accum = 0, 0, 0, 0, 0
        start = time.time();

        hyperparam_dict = {'b_identity': 0.}
        helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')
        hyper_param = np.asarray([global_args.curr_epoch, hyperparam_dict['b_identity']])
        
        random_idx = np.random.randint(0, high=data_loader.curr_max_iter-1)
        for batch_idx, curr_batch_size, batch in data_loader: 
            if random_idx == batch_idx: 
                try:
                    random_batch_data = batch['observed']['data']['image'].copy()
                except: pass
            div_bool, enc_bool, cri_bool, gen_bool = scheduler(global_args.curr_epoch, batch_idx)
            curr_feed_dict = input_dict_func(batch, hyper_param)

            if optimize_bool is None: optimize_bool = np.asarray([div_bool, enc_bool, cri_bool, gen_bool])
            else: optimize_bool[:] = [div_bool, enc_bool, cri_bool, gen_bool] 
            optimize_list_np = sess.run((all_cost_tf[optimize_bool].tolist()+(all_step_tf[optimize_bool].tolist())), feed_dict = curr_feed_dict)
            for i, key in enumerate(all_opt_keys[optimize_bool]): 
                np_costs[key] = optimize_list_np[i]
                step_counts[key] += 1

            train_div_loss_accum += curr_batch_size*np_costs['div']
            train_enc_loss_accum += curr_batch_size*np_costs['enc']
            train_cri_loss_accum += curr_batch_size*np_costs['cri']
            train_gen_loss_accum += curr_batch_size*np_costs['gen']
            train_batch_size_accum += curr_batch_size

            if batch_idx % global_args.log_interval == 0:
                end = time.time();
                report_count = report_count+1

                report = []
                report.append(('Epoch {}', None, global_args.curr_epoch))
                report.append(('[{:7d}]', None, batch_idx * curr_batch_size))
                report.append(('Time {:.2f}', None, (end - start)))
                report.append(('d {:2d}', None, step_counts['div']))
                report.append(('e {:2d}', None, step_counts['enc']))
                report.append(('c {:2d}', None, step_counts['cri']))
                report.append(('g {:2d}', None, step_counts['gen']))
                report.append(['Div {:.2f}', model.div_cost if hasattr(model, 'div_cost') else None, None if hasattr(model, 'div_cost') else 0])
                report.append(['Enc {:.2f}', model.enc_cost if hasattr(model, 'enc_cost') else None, None if hasattr(model, 'enc_cost') else 0])
                report.append(['Cri {:.2f}', model.cri_cost if hasattr(model, 'cri_cost') else None, None if hasattr(model, 'cri_cost') else 0])
                report.append(['Gen {:.2f}', model.gen_cost if hasattr(model, 'gen_cost') else None, None if hasattr(model, 'gen_cost') else 0])
                report.append(['Enc Reg {:.2f}', model.enc_reg_cost if hasattr(model, 'enc_reg_cost') else None, None if hasattr(model, 'enc_reg_cost') else 0])
                report.append(['Cri Reg {:.2f}', model.cri_reg_cost if hasattr(model, 'cri_reg_cost') else None, None if hasattr(model, 'cri_reg_cost') else 0])
                report.append(['C-real {:.2f}', model.mean_critic_real if hasattr(model, 'mean_critic_real') else None, None if hasattr(model, 'mean_critic_real') else 0])
                report.append(['C-reg {:.2f}', model.mean_critic_reg if hasattr(model, 'mean_critic_reg') else None, None if hasattr(model, 'mean_critic_reg') else 0])
                report.append(['C-gen {:.2f}', model.mean_critic_gen if hasattr(model, 'mean_critic_gen') else None, None if hasattr(model, 'mean_critic_gen') else 0])
                report.append(['OT-Prim {:.2f}', model.mean_OT_primal if hasattr(model, 'mean_OT_primal') else None, None if hasattr(model, 'mean_OT_primal') else 0])
                report.append(['OT-Dual {:.2f}', model.mean_OT_dual if hasattr(model, 'mean_OT_dual') else None, None if hasattr(model, 'mean_OT_dual') else 0])
                report.append(['Coupling-pen {:.2f}', model.mean_coupling_line_grad_vector_penalties if hasattr(model, 'mean_coupling_line_grad_vector_penalties') else None, None if hasattr(model, 'mean_coupling_line_grad_vector_penalties') else 0])
                report.append(['Trivial-pen {:.2f}', model.mean_trivial_line_grad_norm_1_penalties if hasattr(model, 'mean_trivial_line_grad_norm_1_penalties') else None, None if hasattr(model, 'mean_trivial_line_grad_norm_1_penalties') else 0])
                report.append(['TTR {:.2f}', model.WAE_WGAN_grad_dist_mean if hasattr(model, 'WAE_WGAN_grad_dist_mean') else None, None if hasattr(model, 'WAE_WGAN_grad_dist_mean') else 0])

                report_format, report_value_list = helper.get_report_formatted(report, sess, curr_feed_dict)
                written_report_1_format, written_report_1_value_list = helper.get_report_formatted(report[7:11], sess, curr_feed_dict)
                written_report_2_format, written_report_2_value_list = helper.get_report_formatted(report[16:], sess, curr_feed_dict)

                print('Train: '+report_format.format(*report_value_list))
                with open(global_args.exp_dir+"training_traces.txt", "a") as text_file:
                    text_file.write(written_report_1_format.format(*written_report_1_value_list)+'\n')

                with open(global_args.exp_dir+"training_primal_dual.txt", "a") as text_file:
                    text_file.write(written_report_2_format.format(*written_report_2_value_list)+'\n')

                start = time.time()
                if global_args.in_between_vis>0 and report_count % global_args.in_between_vis == 0: 
                    # for mode in ['Fixed', 'Random']:
                    for mode in ['Random',]:
                        if mode == 'Fixed': batch['observed']['data']['image'] = fixed_batch_data
                        else: batch['observed']['data']['image'] = random_batch_data
                        curr_feed_dict = input_dict_func(batch, hyper_param)
                        distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, 
                        model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
                        save_dir=global_args.exp_dir+'Visualization/train_'+mode, postfix='train_'+mode+'_'+str(global_args.curr_epoch)+'_e', postfix2='train_'+mode+'_m')
        
        train_div_loss_accum /= train_batch_size_accum
        train_enc_loss_accum /= train_batch_size_accum
        train_cri_loss_accum /= train_batch_size_accum
        train_gen_loss_accum /= train_batch_size_accum

        summary_str = sess.run(merged_summaries, feed_dict = curr_feed_dict)
        summary_writer.add_summary(summary_str, (tf.train.global_step(sess, global_step)))
                   
        if global_args.curr_epoch % global_args.vis_interval == 0:
            report = []
            report.append(['Epoch {}', None, global_args.curr_epoch])
            report.append(['Acc Div Cost {:.2f}', None, train_div_loss_accum])
            report.append(['Acc Enc Cost {:.2f}', None, train_enc_loss_accum])
            report.append(['Acc cri Cost {:.2f}', None, train_cri_loss_accum])
            report.append(['Acc Gen Cost {:.2f}', None, train_gen_loss_accum])
            
            report_format, report_value_list = helper.get_report_formatted(report, sess, curr_feed_dict)
            print('====> Average Train: '+report_format.format(*report_value_list))

            # for mode in ['Fixed', 'Random']:
            for mode in ['Random',]:
                if mode == 'Fixed': batch['observed']['data']['image'] = fixed_batch_data
                else: batch['observed']['data']['image'] = random_batch_data
                curr_feed_dict = input_dict_func(batch, hyper_param)
                distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, 
                model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
                save_dir=global_args.exp_dir+'Visualization/train_'+mode, postfix='train_'+mode+'_'+str(global_args.curr_epoch)+'_e', postfix2='train_'+mode+'_m')

            checkpoint_path1 = global_args.exp_dir+'checkpoint/'
            print('====> Saving checkpoint. Epoch: ', global_args.curr_epoch); start_tmp = time.time()
            if global_args.save_checkpoints: helper.save_checkpoint(saver, sess, global_step, checkpoint_path1) 
            end_tmp = time.time(); print('Checkpoint path: '+checkpoint_path1+'   ====> It took: ', end_tmp - start_tmp)
            if global_args.curr_epoch % 3*global_args.vis_interval == 0: 
                checkpoint_path2 = global_args.exp_dir+'checkpoint2/'
                print('====> Saving checkpoint backup. Epoch: ', global_args.curr_epoch); start_tmp = time.time()
                if global_args.save_checkpoints: helper.save_checkpoint(saver, sess, global_step, checkpoint_path2) 
                end_tmp = time.time(); print('Checkpoint path: '+checkpoint_path2+'   ====> It took: ', end_tmp - start_tmp)
        
    def test():
        global random_batch_data, fixed_batch_data
        test_np_costs = {'div': 0, 'enc': 0, 'cri': 0, 'gen': 0}

        data_loader.eval()
        test_div_loss_accum, test_enc_loss_accum, test_cri_loss_accum, test_gen_loss_accum, \
        test_mean_OT_primal_accum, test_mean_OT_dual_accum, test_batch_size_accum = 0, 0, 0, 0, 0, 0, 0
        start = time.time()

        hyperparam_dict = {'b_identity': 0.}
        helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')
        hyper_param = np.asarray([global_args.curr_epoch, hyperparam_dict['b_identity']])

        random_idx = np.random.randint(0, high=data_loader.curr_max_iter-1)
        for batch_idx, curr_batch_size, batch in data_loader: 
            if random_idx == batch_idx: 
                try:
                    random_batch_data = batch['observed']['data']['image'].copy()
                except: pass

            curr_feed_dict = input_dict_func(batch, hyper_param)       
            
            test_np_costs['div'], test_np_costs['enc'], test_np_costs['cri'], test_np_costs['gen'] = \
            sess.run([model.div_cost, model.enc_cost, model.cri_cost, model.gen_cost], 
            feed_dict = curr_feed_dict)

            test_np_mean_OT_primal, test_np_mean_OT_dual = 0, 0
            if hasattr(model, 'mean_OT_primal'): test_np_mean_OT_primal = sess.run(model.mean_OT_primal, feed_dict = curr_feed_dict)
            if hasattr(model, 'mean_OT_dual'): test_np_mean_OT_dual = sess.run(model.mean_OT_dual, feed_dict = curr_feed_dict)
            
            test_div_loss_accum += curr_batch_size*test_np_costs['div']
            test_enc_loss_accum += curr_batch_size*test_np_costs['enc']
            test_cri_loss_accum += curr_batch_size*test_np_costs['cri']
            test_gen_loss_accum += curr_batch_size*test_np_costs['gen']
            test_mean_OT_primal_accum += curr_batch_size*test_np_mean_OT_primal
            test_mean_OT_dual_accum += curr_batch_size*test_np_mean_OT_dual
            test_batch_size_accum += curr_batch_size

        test_div_loss_accum /= test_batch_size_accum
        test_enc_loss_accum /= test_batch_size_accum
        test_cri_loss_accum /= test_batch_size_accum
        test_gen_loss_accum /= test_batch_size_accum
        test_mean_OT_primal_accum /= test_batch_size_accum
        test_mean_OT_dual_accum /= test_batch_size_accum

        end = time.time();
        report = []
        report.append(['Epoch {}', None, global_args.curr_epoch])
        report.append(['Time {:.2f}', None, (end - start)])
        report.append(['Acc Div Cost {:.2f}', None, test_div_loss_accum])
        report.append(['Acc Enc Cost {:.2f}', None, test_enc_loss_accum])
        report.append(['Acc cri Cost {:.2f}', None, test_cri_loss_accum])
        report.append(['Acc Gen Cost {:.2f}', None, test_gen_loss_accum])
        report.append(['Acc OT-Primal {:.2f}', None, test_mean_OT_primal_accum])
        report.append(['Acc OT-Dual {:.2f}', None, test_mean_OT_dual_accum])

        report_format, report_value_list = helper.get_report_formatted(report, sess, curr_feed_dict)
        written_report_1_format, written_report_1_value_list = helper.get_report_formatted(report[2:6], sess, curr_feed_dict)
        written_report_2_format, written_report_2_value_list = helper.get_report_formatted(report[6:], sess, curr_feed_dict)
                
        print('====> Average Test: '+report_format.format(*report_value_list))
        with open(global_args.exp_dir+"test_traces.txt", "a") as text_file:
                  text_file.write(written_report_1_format.format(*written_report_1_value_list)+'\n')

        with open(global_args.exp_dir+"test_primal_dual.txt", "a") as text_file:
                  text_file.write(written_report_2_format.format(*written_report_2_value_list)+'\n')

        # for mode in ['Fixed', 'Random']:
        for mode in ['Random',]:
            if mode == 'Fixed': batch['observed']['data']['image'] = fixed_batch_data
            else: batch['observed']['data']['image'] = random_batch_data
            curr_feed_dict = input_dict_func(batch, hyper_param)
            distributions.visualizeProductDistribution4(sess, curr_feed_dict, batch, model.input_dist, model.reg_dist, 
            model.reg_target_dist, model.reconst_dist, model.obs_sample_dist, model.gen_obs_sample_dist, 
            save_dir=global_args.exp_dir+'Visualization/test_'+mode, postfix='test_'+mode+'_'+str(global_args.curr_epoch)+'_e', postfix2='test_'+mode+'_m')


    def visualize(mode='train'):
        b_zero_one_range = True

        if mode=='train': data_loader.train(randomize=False)
        else: data_loader.eval()
        hyperparam_dict = {'b_identity': 0.}
        helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')
        all_np_posterior_latent_code= None
        all_np_prior_latent_code = None
        all_np_input_sample = None
        all_np_reconst_sample = None
        all_np_interpolate_sample = None
        all_np_fixed_sample = None
        all_np_fixed_grid_sample = None
        all_labels_np = None

        print('\n*************************************   VISUALIZATION STAGE: '+mode+'   *************************************\n')
        print('Obtaining visualization data.')
        start = time.time();
        
        hyper_param = np.asarray([global_args.curr_epoch, hyperparam_dict['b_identity']])
        for batch_idx, curr_batch_size, batch in data_loader: 
            curr_feed_dict = input_dict_func(batch, hyper_param)    

            if mode=='test' and ((global_args.latent_vis_TSNE_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_TSNE_epoch_rate[0] == global_args.latent_vis_TSNE_epoch_rate[1]) or \
                                 (global_args.latent_vis_UMAP_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_UMAP_epoch_rate[0] == global_args.latent_vis_UMAP_epoch_rate[1])): 
                np_posterior_latent_code, np_prior_latent_code = sess.run([model.posterior_latent_code, model.prior_latent_code], feed_dict = curr_feed_dict)
                if all_np_posterior_latent_code is None: all_np_posterior_latent_code = np_posterior_latent_code
                else: all_np_posterior_latent_code = np.concatenate([all_np_posterior_latent_code, np_posterior_latent_code], axis=0)
                if all_np_prior_latent_code is None: all_np_prior_latent_code = np_prior_latent_code
                else: all_np_prior_latent_code = np.concatenate([all_np_prior_latent_code, np_prior_latent_code], axis=0)

            if (global_args.reconst_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.reconst_vis_epoch_rate[0] == global_args.reconst_vis_epoch_rate[1]): 
                if all_np_input_sample is None or all_np_input_sample.shape[0]<400:
                    np_input_sample = sess.run(model.input_sample['image'], feed_dict = curr_feed_dict)
                    if all_np_input_sample is None: all_np_input_sample = np_input_sample
                    else: all_np_input_sample = np.concatenate([all_np_input_sample, np_input_sample], axis=0)
                if all_np_reconst_sample is None or all_np_reconst_sample.shape[0]<400:
                    np_reconst_sample = sess.run(model.reconst_sample['image'], feed_dict = curr_feed_dict)
                    if all_np_reconst_sample is None: all_np_reconst_sample = np_reconst_sample
                    else: all_np_reconst_sample = np.concatenate([all_np_reconst_sample, np_reconst_sample], axis=0)

            if (global_args.interpolate_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.interpolate_vis_epoch_rate[0] == global_args.interpolate_vis_epoch_rate[1]): 
                if all_np_interpolate_sample is None or all_np_interpolate_sample.shape[0]<20:
                    _, np_interpolated_obs = sess.run([model.interpolated_posterior_latent_code, model.interpolated_obs['image']], feed_dict = curr_feed_dict)
                    if all_np_interpolate_sample is None: all_np_interpolate_sample = np_interpolated_obs
                    else: all_np_interpolate_sample = np.concatenate([all_np_interpolate_sample, np_interpolated_obs], axis=0)

            if (global_args.fixed_samples_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.fixed_samples_vis_epoch_rate[0] == global_args.fixed_samples_vis_epoch_rate[1]): 
                if all_np_fixed_sample is None or all_np_fixed_sample.shape[0]<400:
                    np_constant_obs_sample = sess.run(model.constant_obs_sample['image'], feed_dict = curr_feed_dict)
                    if all_np_fixed_sample is None: all_np_fixed_sample = np_constant_obs_sample
                    else: all_np_fixed_sample = np.concatenate([all_np_fixed_sample, np_constant_obs_sample], axis=0)

                if global_args.n_latent == 2 and (all_np_fixed_grid_sample is None or all_np_fixed_grid_sample.shape[0]<400):
                    np_constant_obs_grid_sample = sess.run(model.constant_obs_grid_sample['image'], feed_dict = curr_feed_dict)
                    if all_np_fixed_grid_sample is None: all_np_fixed_grid_sample = np_constant_obs_grid_sample
                    else: all_np_fixed_grid_sample = np.concatenate([all_np_fixed_grid_sample, np_constant_obs_grid_sample], axis=0)

            if batch['context']['data']['flat'] is not None:
              if all_labels_np is None: all_labels_np = batch['context']['data']['flat'][:,0,:]
              else: all_labels_np = np.concatenate([all_labels_np, batch['context']['data']['flat'][:,0,:]], axis=0)
        end = time.time();
        print('Obtained visualization data: Time: {:.3f}'.format((end - start)))

        if mode=='test' and ((global_args.pigeonhole_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.pigeonhole_score_epoch_rate[0] == global_args.pigeonhole_score_epoch_rate[1]) or \
                             (global_args.fid_inception_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.fid_inception_score_epoch_rate[0] == global_args.fid_inception_score_epoch_rate[1])):
            n_random_samples = 50000
            print('Obtaining {:d} random samples.'.format(n_random_samples))
            start = time.time();
            random_samples_from_model = np.zeros((n_random_samples, 1, data_loader.image_size, data_loader.image_size, 3))
            curr_index = 0 
            while curr_index<n_random_samples:
                curr_samples = sess.run(model.gen_obs_sample['image'], feed_dict = curr_feed_dict)
                end_index = min(curr_index+curr_samples.shape[0], n_random_samples)
                random_samples_from_model[curr_index:end_index, ...] = curr_samples[:end_index-curr_index,...]
                curr_index = end_index
            end = time.time()
            print('Obtained random samples: Time: {:.3f}'.format((end - start)))
            
            if mode=='test' and global_args.pigeonhole_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.pigeonhole_score_epoch_rate[0] == global_args.pigeonhole_score_epoch_rate[1]: 
                print('Computing pidgeon-hole score.')
                start = time.time();
                pigeonhole_mean, pigeonhole_std = helper.pigeonhole_score(random_samples_from_model, subset=500, neigh=0.05)
                end = time.time()
                print('Pidgeon-hole Score -- Time: {:.3f} Mean: {:.3f} Std: {:.3f}'.format((end - start), pigeonhole_mean, pigeonhole_std))
                with open(global_args.exp_dir+mode+"_pidgeonhole_score.txt", "a") as text_file:
                    text_file.write("Epoch: {:d} Mean: {:.3f} Std: {:.3f}\n".format(global_args.curr_epoch, pigeonhole_mean, pigeonhole_std)+'\n')

            if mode=='test' and global_args.fid_inception_score_epoch_rate[0]>0 and global_args.curr_epoch % global_args.fid_inception_score_epoch_rate[0] == global_args.fid_inception_score_epoch_rate[1]:
                print('Computing inception stats.')
                start = time.time();
                model_fid_value, model_inception_mean, model_inception_std, real_inception_mean, real_inception_std = \
                    InceptionScoreModel.fid_and_inception_score(random_samples_from_model, dataset_to_use, data_loader)
                end = time.time()
                print('Inception Stats -- Time: {:.3f} FID: {:.3f} Inc. Mean: {:.3f} Inc. Std: {:.3f} Real Inc. Mean: {:.3f} Real Inc. Std: {:.3f}'.format(\
                    (end - start), model_fid_value, model_inception_mean, model_inception_std, real_inception_mean, real_inception_std))
                with open(global_args.exp_dir+mode+"_inception_stats.txt", "a") as text_file:
                    text_file.write('Epoch: {:d} FID: {:.3f} Inc. Mean: {:.3f} Inc. Std: {:.3f} Real Inc. Mean: {:.3f} Real Inc. Std: {:.3f}'.format(\
                        global_args.curr_epoch, model_fid_value, model_inception_mean, model_inception_std, real_inception_mean, real_inception_std)+'\n')

        if global_args.reconst_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.reconst_vis_epoch_rate[0] == global_args.reconst_vis_epoch_rate[1]: 
            print('Visualizing reconstructions')
            start = time.time();
            if b_zero_one_range: np.clip(all_np_reconst_sample, 0, 1, out=all_np_reconst_sample) 
            all_sample_reconst = helper.interleave_data([all_np_input_sample, all_np_reconst_sample])
            helper.visualize_images2(all_sample_reconst[:2*int(np.sqrt(all_np_input_sample.shape[0]))**2, ...], 
            block_size=[int(np.sqrt(all_np_input_sample.shape[0])), 2*int(np.sqrt(all_np_input_sample.shape[0]))], 
            save_dir=global_args.exp_dir+'Visualization/'+mode+'_real_sample_reconst/', postfix = '_'+mode+'_real_sample_reconst_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_real_sample_reconst'+'_m')
            
            end = time.time()
            print('Visualized reconstructions: Time: {:.3f}'.format((end - start)))

        if global_args.interpolate_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.interpolate_vis_epoch_rate[0] == global_args.interpolate_vis_epoch_rate[1]: 
            print('Visualizing interpolations')
            start = time.time();
            if b_zero_one_range: np.clip(all_np_interpolate_sample, 0, 1, out=all_np_interpolate_sample) 
            helper.visualize_images2(all_np_interpolate_sample.reshape(-1, 1, *all_np_interpolate_sample.shape[2:]), 
            block_size=[all_np_interpolate_sample.shape[0], all_np_interpolate_sample.shape[1]], 
            save_dir=global_args.exp_dir+'Visualization/'+mode+'_real_sample_interpolated/', postfix = '_'+mode+'_real_sample_interpolated_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_real_sample_interpolated'+'_m')
            
            end = time.time()
            print('Visualized interpolations: Time: {:.3f}'.format((end - start)))

        if mode=='test' and global_args.fixed_samples_vis_epoch_rate[0]>0 and global_args.curr_epoch % global_args.fixed_samples_vis_epoch_rate[0] == global_args.fixed_samples_vis_epoch_rate[1]: 
            print('Visualizing fixed samples')
            start = time.time();
            if b_zero_one_range: np.clip(all_np_fixed_sample, 0, 1, out=all_np_fixed_sample) 
            helper.visualize_images2(all_np_fixed_sample[:int(np.sqrt(all_np_fixed_sample.shape[0]))**2, ...], 
            block_size=[int(np.sqrt(all_np_fixed_sample.shape[0])), int(np.sqrt(all_np_fixed_sample.shape[0]))], 
            save_dir=global_args.exp_dir+'Visualization/'+mode+'_fixed_sample/', postfix = '_'+mode+'_fixed_sample_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_fixed_sample'+'_m')

            if global_args.n_latent == 2:
                if b_zero_one_range: np.clip(all_np_fixed_grid_sample, 0, 1, out=all_np_fixed_grid_sample) 
                helper.visualize_images2(all_np_fixed_grid_sample[:int(np.sqrt(all_np_fixed_grid_sample.shape[0]))**2, ...], 
                block_size=[int(np.sqrt(all_np_fixed_grid_sample.shape[0])), int(np.sqrt(all_np_fixed_grid_sample.shape[0]))], 
                save_dir=global_args.exp_dir+'Visualization/'+mode+'_fixed_sample/', postfix = '_'+mode+'_fixed_grid_sample_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_fixed_grid_sample'+'_m')
            
            end = time.time()
            print('Visualized fixed samples: Time: {:.3f}'.format((end - start)))

        if mode=='test' and ((global_args.latent_vis_TSNE_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_TSNE_epoch_rate[0] == global_args.latent_vis_TSNE_epoch_rate[1]) or \
                             (global_args.latent_vis_UMAP_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_UMAP_epoch_rate[0] == global_args.latent_vis_UMAP_epoch_rate[1])): 
            
            # n_vis_samples = int(all_np_prior_latent_code.shape[0]*0.4)
            n_vis_samples = min(10000, all_np_posterior_latent_code.shape[0])
            all_np_prior_latent_code = all_np_prior_latent_code[:n_vis_samples, :]

            chosen_indeces = np.random.permutation(np.arange(all_np_posterior_latent_code.shape[0]))
            chosen_indeces = chosen_indeces[:n_vis_samples]
            all_np_posterior_latent_code = all_np_posterior_latent_code[chosen_indeces, :]
            if all_labels_np is not None: all_labels_np = all_labels_np[chosen_indeces, :]

            all_input = np.concatenate([all_np_posterior_latent_code, all_np_prior_latent_code], axis=0)
            chosen_indeces2 = np.random.permutation(np.arange(all_input.shape[0]))
            inverse_chosen_indeces2 = np.zeros((all_input.shape[0],), dtype=int)
            for ind, e in enumerate(chosen_indeces2): inverse_chosen_indeces2[e] = ind

            if (global_args.latent_vis_TSNE_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_TSNE_epoch_rate[0] == global_args.latent_vis_TSNE_epoch_rate[1]):
                print('Visualizing latents - TSNE.')
                start = time.time();

                if global_args.n_latent == 2:
                    helper.dataset_plotter([all_np_posterior_latent_code,], colors=['g',], point_thickness = 10, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_posterior/', postfix = '_'+mode+'_TSNE_posterior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_posterior'+'_m')
                    helper.dataset_plotter([all_np_prior_latent_code,], colors=['r',], point_thickness = 10, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_prior/', postfix = '_'+mode+'_TSNE_prior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_prior'+'_m')
                    helper.dataset_plotter([all_np_prior_latent_code, all_np_posterior_latent_code], point_thickness = 10, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_prior_posterior/', postfix = '_'+mode+'_TSNE_prior_posterior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_prior_posterior'+'_m')

                    rand_indices_TSNE = np.arange(all_np_posterior_latent_code.shape[0]).astype(np.int)
                    np.random.permutation(rand_indices_TSNE)
                    rand_indices_TSNE = rand_indices_TSNE[:global_args.batch_size].astype(np.int)
                    pdb.set_trace()
                    helper.dataset_plotter([all_np_posterior_latent_code[rand_indices_TSNE, ...],], colors=['g',], point_thickness = 10, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_posterior_batch/', postfix = '_'+mode+'_TSNE_posterior_batch_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_posterior_batch'+'_m')
                    helper.dataset_plotter([all_np_prior_latent_code[rand_indices_TSNE, ...],], colors=['r',], point_thickness = 10, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_prior_batch/', postfix = '_'+mode+'_TSNE_prior_batch_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_prior_batch'+'_m')
                    helper.dataset_plotter([all_np_prior_latent_code[rand_indices_TSNE, ...], all_np_posterior_latent_code[rand_indices_TSNE, ...]], point_thickness = 10, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_prior_posterior_batch/', postfix = '_'+mode+'_TSNE_prior_posterior_batch_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_prior_posterior_batch'+'_m')
                else:
                    all_tsne_scrambled = TSNE().fit_transform(all_input[chosen_indeces2, :])
                    all_tsne = all_tsne_scrambled[inverse_chosen_indeces2, :]

                    all_tsne_centered = all_tsne-np.mean(all_tsne, axis=0)[np.newaxis, :]
                    all_tsne_normalized = all_tsne_centered/(np.std(all_tsne_centered, axis=0)[np.newaxis, :]+1e-7)
                    tsne_normalized_posterior = all_tsne_normalized[:all_np_posterior_latent_code.shape[0],:]
                    tsne_normalized_prior = all_tsne_normalized[all_np_posterior_latent_code.shape[0]:,:]

                    tsne_class_wise_posterior = []
                    tsne_class_wise_sizes = []
                    if all_labels_np is None:
                      tsne_class_wise_posterior.append(tsne_normalized_posterior)
                    else:
                      for c in range(all_labels_np.shape[1]):
                        tsne_curr_class_posterior = tsne_normalized_posterior[all_labels_np[:,c].astype(bool),:]
                        tsne_class_wise_sizes.append(tsne_curr_class_posterior.shape[0])
                        tsne_class_wise_posterior.append(tsne_curr_class_posterior)
                      for i in range(len(tsne_class_wise_posterior)):
                        tsne_class_wise_posterior[i] = tsne_class_wise_posterior[i][:min(tsne_class_wise_sizes),:]

                    helper.dataset_plotter(tsne_class_wise_posterior, point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_posterior_classwise/', postfix = '_'+mode+'_TSNE_posterior_classwise_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_posterior_classwise'+'_m')
                    helper.dataset_plotter([tsne_normalized_posterior,], colors=['r',], point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_posterior/', postfix = '_'+mode+'_TSNE_posterior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_posterior'+'_m')
                    helper.dataset_plotter([tsne_normalized_prior,], colors=['g',], point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_prior/', postfix = '_'+mode+'_TSNE_prior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_prior'+'_m')
                    helper.dataset_plotter([tsne_normalized_posterior, tsne_normalized_prior], point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_TSNE_prior_posterior/', postfix = '_'+mode+'_TSNE_prior_posterior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_TSNE_prior_posterior'+'_m')
                    
                    end = time.time()
                    print('Visualized latents - TSNE: Time: {:.3f}\n'.format((end - start)))

            if (global_args.latent_vis_UMAP_epoch_rate[0]>0 and global_args.curr_epoch % global_args.latent_vis_UMAP_epoch_rate[0] == global_args.latent_vis_UMAP_epoch_rate[1]):
                print('Visualizing latents - UMAP.')
                start = time.time();

                if not global_args.n_latent == 2:
                    all_umap_scrambled = UMAP().fit_transform(all_input[chosen_indeces2, :])
                    all_umap = all_umap_scrambled[inverse_chosen_indeces2, :]

                    all_umap_centered = all_umap-np.mean(all_umap, axis=0)[np.newaxis, :]
                    all_umap_normalized = all_umap_centered/(np.std(all_umap_centered, axis=0)[np.newaxis, :]+1e-7)
                    umap_normalized_posterior = all_umap_normalized[:all_np_posterior_latent_code.shape[0],:]
                    umap_normalized_prior = all_umap_normalized[all_np_posterior_latent_code.shape[0]:,:]

                    umap_class_wise_posterior = []
                    umap_class_wise_sizes = []
                    if all_labels_np is None:
                      umap_class_wise_posterior.append(umap_normalized_posterior)
                    else:
                      for c in range(all_labels_np.shape[1]):
                        umap_curr_class_posterior = umap_normalized_posterior[all_labels_np[:,c].astype(bool),:]
                        umap_class_wise_sizes.append(umap_curr_class_posterior.shape[0])
                        umap_class_wise_posterior.append(umap_curr_class_posterior)
                      for i in range(len(umap_class_wise_posterior)):
                        umap_class_wise_posterior[i] = umap_class_wise_posterior[i][:min(umap_class_wise_sizes),:]

                    helper.dataset_plotter(umap_class_wise_posterior, point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_UMAP_posterior_classwise/', postfix = '_'+mode+'_UMAP_posterior_classwise_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_UMAP_posterior_classwise'+'_m')
                    helper.dataset_plotter([umap_normalized_posterior,], colors=['r',], point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_UMAP_posterior/', postfix = '_'+mode+'_UMAP_posterior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_UMAP_posterior'+'_m')
                    helper.dataset_plotter([umap_normalized_prior,], colors=['g',], point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_UMAP_prior/', postfix = '_'+mode+'_UMAP_prior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_UMAP_prior'+'_m')
                    helper.dataset_plotter([umap_normalized_posterior, umap_normalized_prior], point_thickness = 4, save_dir = global_args.exp_dir+'Visualization/'+mode+'_UMAP_prior_posterior/', postfix = '_'+mode+'_UMAP_prior_posterior_'+str(global_args.curr_epoch)+'_e', postfix2 = '_'+mode+'_UMAP_prior_posterior'+'_m')
                    
                end = time.time()
                print('Visualized latents - UMAP: Time: {:.3f}\n'.format((end - start)))

    print('Starting training.')
    while global_args.curr_epoch < global_args.epochs + 1:
        train()
        if global_args.test_epoch_rate[0]>0 and global_args.curr_epoch % global_args.test_epoch_rate[0] == global_args.test_epoch_rate[1]: 
            test()
        visualize(mode='train')
        visualize(mode='test')
        print('Experiment Directory: ', global_args.exp_dir)

        global_args.curr_epoch += 1            
        gc.collect()









            # if model.__module__ == 'models.WAE.Model':
            #     if div_step_tf is not None:
            #         np_div_step, np_enc_step, np_gen_step, np_costs['div'], np_costs['enc'], np_costs['gen'], np_costs['cri'] = \
            #         sess.run([div_step_tf, enc_step_tf, gen_step_tf, model.div_cost, model.enc_cost, 
            #         model.cri_cost, model.gen_cost], feed_dict = curr_feed_dict)
            #         step_counts['div'] = step_counts['div']+1
            #         step_counts['enc'] = step_counts['enc']+1
            #         step_counts['gen'] = step_counts['gen']+1
            #     else:
            #         np_enc_step, np_gen_step, np_costs['div'], np_costs['enc'], np_costs['gen'], np_costs['cri'] = \
            #         sess.run([enc_step_tf, gen_step_tf, model.div_cost, model.enc_cost, 
            #         model.cri_cost, model.gen_cost], feed_dict = curr_feed_dict)
            #         step_counts['enc'] = step_counts['enc']+1
            #         step_counts['gen'] = step_counts['gen']+1    
            # else:
            #     if (div_bool and div_step_tf is not None) and (enc_bool and enc_step_tf is not None) and (cri_bool and cri_step_tf is not None):
            #         np_div_step, np_enc_step, np_cri_step = sess.run([div_step_tf, enc_step_tf, cri_step_tf], feed_dict = curr_feed_dict)
            #         step_counts['div'] = step_counts['div']+1
            #         step_counts['enc'] = step_counts['enc']+1
            #         step_counts['cri'] = step_counts['cri']+1
            #     else:   
            #         if div_bool and div_step_tf is not None:
            #             np_div_step = sess.run(div_step_tf, feed_dict = curr_feed_dict)
            #             step_counts['div'] = step_counts['div']+1

            #         if (enc_bool and enc_step_tf is not None) and (cri_bool and cri_step_tf is not None):
            #             np_enc_step, np_cri_step = sess.run([enc_step_tf, cri_step_tf], feed_dict = curr_feed_dict)
            #             step_counts['enc'] = step_counts['enc']+1
            #             step_counts['cri'] = step_counts['cri']+1
            #         else:
            #             if enc_bool and enc_step_tf is not None:
            #                 np_enc_step = sess.run(enc_step_tf, feed_dict = curr_feed_dict)
            #                 step_counts['enc'] = step_counts['enc']+1
                            
            #             if cri_bool and cri_step_tf is not None:
            #                 np_cri_step = sess.run(cri_step_tf, feed_dict = curr_feed_dict)
            #                 step_counts['cri'] = step_counts['cri']+1

            #     if gen_bool and gen_step_tf is not None:
            #         np_gen_step, np_costs['div'], np_costs['enc'], np_costs['gen'], np_costs['cri'] = \
            #         sess.run([gen_step_tf, model.div_cost, model.enc_cost, 
            #         model.cri_cost, model.gen_cost], feed_dict = curr_feed_dict)
            #         step_counts['gen'] = step_counts['gen']+1














# if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
#                 helper.visualize_datasets(sess, input_dict_func(batch), data_loader.dataset, generative_dict['obs_sample_out'],
#                                           generative_dict['latent_sample_out'], train_outs_dict['transport_sample'], train_outs_dict['input_sample'],
#                                           save_dir=global_args.exp_dir+'Visualization/', postfix=str(epoch)) 

#                 xmin, xmax, ymin, ymax, X_dense, Y_dense = -2.5, 2.5, -2.5, 2.5, 250, 250
#                 xlist = np.linspace(xmin, xmax, X_dense)
#                 ylist = np.linspace(ymin, ymax, Y_dense)
#                 X, Y = np.meshgrid(xlist, ylist)
#                 XY = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)

#                 batch['observed']['data']['flat'] = XY[:, np.newaxis, :]
#                 cri_cost_real_np = sess.run(train_outs_dict['critic_real'], feed_dict = input_dict_func(batch, hyper_param))

#                 batch['observed']['data']['flat'] = data_loader.dataset[:, np.newaxis, :]
#                 cri_cost_real_real_np = sess.run(train_outs_dict['critic_real'], feed_dict = input_dict_func(batch, hyper_param))

#                 cri_mean = cri_cost_real_real_np.mean()
#                 cri_std = cri_cost_real_real_np.std()
#                 cri_max = cri_mean+2*cri_std
#                 cri_min = cri_mean-2*cri_std

#                 np.clip(cri_cost_real_np, cri_min, cri_max, out=cri_cost_real_np)
#                 f = np.reshape(cri_cost_real_np[:,0,0], [Y_dense, X_dense])
#                 helper.plot_ffs(X, Y, f, save_dir=global_args.exp_dir+'Visualization/cririminator_function/', postfix='cririminator_function'+str(epoch))
                
#             else:



        # no_gen_epoch_rate, no_gen_epoch_num = 20, 0
        # if t<3 or ((epoch%no_gen_epoch_rate>=no_gen_epoch_num) and t%(critic_rate*generator_rate)==0): gen_bool = True



    # def visualize(epoch):
    #     data_loader.eval()

    #     hyperparam_dict = {'b_identity': 0.}
    #     helper.update_dict_from_file(hyperparam_dict, './hyperparam_file.py')
    #     all_np_posterior_latent_code= None
    #     all_np_prior_latent_code = None
    #     all_np_input_sample = None
    #     all_np_reconst_sample = None
    #     all_labels_np = None

    #     print('\n*************************************   VISUALIZATION STAGE   *************************************\n')
    #     print('Obtaining visualization data.')
    #     start = time.time();
    #     for batch_idx, curr_batch_size, batch in data_loader: 
    #         hyper_param = np.asarray([epoch, hyperparam_dict['b_identity']])
    #         curr_feed_dict = input_dict_func(batch, hyper_param)    

    #         if global_args.curr_epoch % global_args.latent_vis_TSNE_epoch_rate[0] == global_args.latent_vis_TSNE_epoch_rate[1]: 
    #             np_posterior_latent_code, np_prior_latent_code = sess.run([model.posterior_latent_code, model.prior_latent_code], feed_dict = curr_feed_dict)
    #             if all_np_posterior_latent_code is None: all_np_posterior_latent_code = np_posterior_latent_code
    #             else: all_np_posterior_latent_code = np.concatenate([all_np_posterior_latent_code, np_posterior_latent_code], axis=0)
    #             if all_np_prior_latent_code is None: all_np_prior_latent_code = np_prior_latent_code
    #             else: all_np_prior_latent_code = np.concatenate([all_np_prior_latent_code, np_prior_latent_code], axis=0)

    #         if global_args.curr_epoch % global_args.reconst_vis_epoch_rate[0] == global_args.reconst_vis_epoch_rate[1]: 
    #             np_input_sample, np_reconst_sample = sess.run([model.input_sample['image'], model.reconst_sample['image']], feed_dict = curr_feed_dict)
    #             if all_np_input_sample is None or all_np_input_sample.shape[0]<400:
    #                 if all_np_input_sample is None: all_np_input_sample = np_input_sample
    #                 else: all_np_input_sample = np.concatenate([all_np_input_sample, np_input_sample], axis=0)
    #             if all_np_reconst_sample is None or all_np_reconst_sample.shape[0]<400:
    #                 if all_np_reconst_sample is None: all_np_reconst_sample = np_reconst_sample
    #                 else: all_np_reconst_sample = np.concatenate([all_np_reconst_sample, np_reconst_sample], axis=0)

    #         if batch['context']['data']['flat'] is not None:
    #           if all_labels_np is None: all_labels_np = batch['context']['data']['flat'][:,0,:]
    #           else: all_labels_np = np.concatenate([all_labels_np, batch['context']['data']['flat'][:,0,:]], axis=0)
    #     end = time.time();
    #     print('Obtained visualization data: Time: {:.3f}'.format((end - start)))

    #     if global_args.compute_inception_score or global_args.compute_pigeonhole_score: 
    #         n_random_samples = 50000
    #         print('Obtaining {:d} random samples.'.format(n_random_samples))
    #         start = time.time();
    #         random_samples_from_model = np.zeros((n_random_samples, 1, data_loader.image_size, data_loader.image_size, 3))
    #         curr_index = 0 
    #         while curr_index<n_random_samples:
    #             curr_samples = sess.run(model.gen_obs_sample['image'], feed_dict = curr_feed_dict)
    #             end_index = min(curr_index+curr_samples.shape[0], n_random_samples)
    #             random_samples_from_model[curr_index:end_index, ...] = curr_samples[:end_index-curr_index,...]
    #             curr_index = end_index
    #         end = time.time()
    #         print('Obtained random samples: Time: {:.3f}'.format((end - start)))
            
    #         if global_args.compute_pigeonhole_score: 
    #             print('Computing pidgeon-hole score.')
    #             start = time.time();
    #             pigeonhole_mean, pigeonhole_std = helper.pigeonhole_score(random_samples_from_model, subset=500, neigh=0.05)
    #             end = time.time()
    #             print('Pidgeon-hole Score -- Time: {:.3f} Mean: {:.3f} Std: {:.3f}'.format((end - start), pigeonhole_mean, pigeonhole_std))
    #         if global_args.compute_inception_score:
    #             print('Computing inception score.')
    #             start = time.time();
    #             inception_mean, inception_std = InceptionScoreModel.inception_score(random_samples_from_model)
    #             end = time.time()
    #             print('Inception Score -- Time: {:.3f} Mean: {:.3f} Std: {:.3f}'.format((end - start), inception_mean, inception_std))


    #     if global_args.curr_epoch % global_args.reconst_vis_epoch_rate[0] == global_args.reconst_vis_epoch_rate[1]: 
    #         print('Visualizing reconstructions')
    #         start = time.time();
    #         all_sample_reconst = helper.interleave_data([all_np_input_sample, all_np_reconst_sample])
    #         helper.visualize_images2(all_sample_reconst[:2*int(np.sqrt(all_np_input_sample.shape[0]))**2, ...], 
    #         block_size=[int(np.sqrt(all_np_input_sample.shape[0])), 2*int(np.sqrt(all_np_input_sample.shape[0]))], 
    #         save_dir=global_args.exp_dir+'Visualization/test_real_sample_reconst/', postfix = '_test_real_sample_reconst'+str(epoch))
            
    #         end = time.time()
    #         print('Visualized reconstructions: Time: {:.3f}'.format((end - start)))

    #     if global_args.curr_epoch % global_args.latent_vis_TSNE_epoch_rate[0] == global_args.latent_vis_TSNE_epoch_rate[1]: 
    #         print('Visualizing latents.')
    #         start = time.time();
    #         # all_np_prior_latent_code = all_np_prior_latent_code[:int(all_np_prior_latent_code.shape[0]*0.4), :]
    #         all_np_prior_latent_code = all_np_prior_latent_code[:2000, :]
    #         all_tsne_input = np.concatenate([all_np_posterior_latent_code, all_np_prior_latent_code], axis=0)


    #         all_tsne = TSNE().fit_transform(all_tsne_input)

    #         all_tsne_centered = all_tsne-np.mean(all_tsne, axis=0)[np.newaxis, :]
    #         all_tsne_normalized = all_tsne_centered/(np.std(all_tsne_centered, axis=0)[np.newaxis, :]+1e-7)
    #         all_tsne_normalized_posterior = all_tsne_normalized[:all_np_posterior_latent_code.shape[0],:]
    #         all_tsne_normalized_prior = all_tsne_normalized[all_np_posterior_latent_code.shape[0]:,:]

    #         class_wise = []
    #         class_wise_sizes = []
    #         if all_labels_np is None:
    #           class_wise.append(all_tsne_normalized_posterior)
    #         else:
    #           for c in range(all_labels_np.shape[1]):
    #             class_wise_z = all_tsne_normalized_posterior[all_labels_np[:,c].astype(bool),:]
    #             class_wise_sizes.append(class_wise_z.shape[0])
    #             class_wise.append(class_wise_z)
    #           for i in range(len(class_wise)):
    #             class_wise[i] = class_wise[i][:min(class_wise_sizes),:]

    #         helper.dataset_plotter(class_wise, save_dir = global_args.exp_dir+'Visualization/z_projection_posterior/', postfix = '_z_projection_posterior'+str(epoch), postfix2 = 'z_projection_posterior')
    #         helper.dataset_plotter([all_tsne_normalized_prior,], save_dir = global_args.exp_dir+'Visualization/z_projection_prior/', postfix = '_z_projection_prior'+str(epoch), postfix2 = 'z_projection_prior')
    #         helper.dataset_plotter([all_tsne_normalized_prior, all_tsne_normalized_posterior], save_dir = global_args.exp_dir+'Visualization/z_projection_prior_posterior/', postfix = '_z_projection_prior_posterior'+str(epoch), postfix2 = 'z_projection_prior_posterior')
            
    #         end = time.time()
    #         print('Visualized latents: Time: {:.3f}\n'.format((end - start)))




# shutil.copyfile('./models/SLVM.py', global_args.exp_dir+'SLVM.py')
# shutil.copyfile('./models/ModelGTM.py', global_args.exp_dir+'ModelGTM.py')
# if os.path.exists('/var/scratch/mcgemici/'): 
#     temp_dir = '/var/scratch/mcgemici/experiments/'+global_args.exp_dir
#     shutil.move(global_args.exp_dir, temp_dir) 
#     os.symlink(os.path.abspath(temp_dir), os.path.abspath(global_args.exp_dir))
#     global_args.exp_dir = temp_dir


    # def scheduler(epoch, t):
    #     gen_bool, cri_bool, trans_bool = False, False, False

    #     if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
    #         if epoch < 100: critic_rate, generator_rate = 2, 5
    #         else: critic_rate, generator_rate = 2, 5
    #         no_gen_epoch_rate, no_gen_epoch_num = 20, 4

    #         if t<3 or t%1==0: trans_bool = True
    #         if t<3 or t%critic_rate==0: cri_bool = True
    #         if t<3 or ((epoch%no_gen_epoch_rate>=no_gen_epoch_num) and t%(critic_rate*generator_rate)==0): gen_bool = True

    #     else:
    #         if epoch < 100: critic_rate, generator_rate = 1, 5
    #         else: critic_rate, generator_rate = 1, 5

    #         no_gen_epoch_rate, no_gen_epoch_num = 20, 0

    #         if t<3 or t%1==0: trans_bool = True
    #         if t<3 or t%critic_rate==0: cri_bool = True
    #         if t<3 or ((epoch%no_gen_epoch_rate>=no_gen_epoch_num) and t%(critic_rate*generator_rate)==0): gen_bool = True

    #     return gen_bool, cri_bool, trans_bool  



# helper.draw_bar_plot(convex_mask_np, y_min_max = [0,1], save_dir=global_args.exp_dir+'Visualization/convex_mask/', postfix='convex_mask'+str(epoch))

    # if global_args.optimizer_class == 'RmsProp':
    #     if data_loader.__module__ == 'datasetLoaders.RandomManifoldDataLoader' or data_loader.__module__ == 'datasetLoaders.ToyDataLoader':
    #         generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                   loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         cririminator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.cririminator_cost, var_list=cririminator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         if model.encoder_cost is not None:
    #             transport_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.encoder_cost, var_list=encoder_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #     else:
    #         generator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.9), 
    #                                   loss=model.generator_cost, var_list=generator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         cririminator_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.cririminator_cost, var_list=cririminator_vars, global_step=global_step, clip_param=global_args.gradient_clipping)
    #         if model.encoder_cost is not None:
    #             transport_step_tf = helper.clipped_optimizer_minimize(optimizer=tf.train.RMSPropOptimizer(learning_rate=global_args.learning_rate, momentum=0.5), 
    #                                       loss=model.encoder_cost, var_list=encoder_vars, global_step=global_step, clip_param=global_args.gradient_clipping)

        
        # helper.visualize_images2(all_np_reconst_sample[:int(np.sqrt(all_np_reconst_sample.shape[0]))**2, ...],
        # block_size=[int(np.sqrt(all_np_reconst_sample.shape[0])), int(np.sqrt(all_np_reconst_sample.shape[0]))],
        # save_dir=global_args.exp_dir+'Visualization/test_reconstruction_only/', postfix='_test_reconstruction_only')

        # helper.visualize_images2(all_np_input_sample[:int(np.sqrt(all_np_input_sample.shape[0]))**2, ...],
        # block_size=[int(np.sqrt(all_np_input_sample.shape[0])), int(np.sqrt(all_np_input_sample.shape[0]))],
        # save_dir=global_args.exp_dir+'Visualization/test_real_sample_only/', postfix='_test_real_sample_only')

# 'critic_reg_mode': ['Uniform Lipschitz',], 'enc_reg_strength': 10, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 20, 'cri_reg_strength': 1}
# 'critic_reg_mode': ['Coupling Gradient Vector','Trivial Lipschitz'], 'enc_reg_strength': 10, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 20, 'cri_reg_strength': 1}
# 'critic_reg_mode': ['Coupling Gradient Vector', 'Uniform Lipschitz'], 'enc_reg_strength': 10, 'enc_inv_MMD_n_trans': 5, 'enc_inv_MMD_strength': 20, 'cri_reg_strength': 1}
