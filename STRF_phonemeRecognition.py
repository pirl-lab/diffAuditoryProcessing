import argparse
import glob
import time
import librosa
import pickle
from os import mkdir
from os.path import join

import numpy as np
from math import floor, ceil
import torch
from torch.utils import data
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap, config, random
config.update("jax_enable_x64", True)

import optax
import flax.linen as nn
from random import random
import colorednoise

from tqdm import tqdm

from strfpy import *
from strfpy_jax import *
from supervisedSTRF import * 
print(f"Training on {jax.default_backend()}...")

import pickle
from functools import partial

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_strfs', required=True, type=int) 
  parser.add_argument('--strf_seed', default=0, type=int) 
  parser.add_argument('--strf_init_method', default='random', type=str, help='random or log. log currently only supports nm_strfs=40') 
  parser.add_argument('--spec_type', required=True, type=str, help='3 options: logMel, logAud, linAud.')
  parser.add_argument('--n_phones', required=True, type=int)
  parser.add_argument('--decoder_type', required=True, type=str, help="whether mlp or cnn")
  parser.add_argument('--encoder_type', required=True, type=str, help="whether strf or cnn")
  parser.add_argument('--input_type', default='spec', type=str, help="audio or spec")
  parser.add_argument('--compression_method', default='identity', type=str, help="identity or power")
  parser.add_argument('--update_lin', action='store_true')
  parser.add_argument('--use_class', action='store_true')

  parser.add_argument('--conv_feats', required=True, nargs="+", type=int)
  parser.add_argument('--pooling_stride', default=2, type=int)
  parser.add_argument('--update_sr', default=1, type=int, help="1 or 0, whether sr is fixed or updated")
  parser.add_argument('--loss_fct', required=True, type=str, help='ctc or xe')
  parser.add_argument('--lr_v', default=0.001, type=float)
  parser.add_argument('--lr_sr', default=0.001, type=float)
  parser.add_argument('--num_steps', default=200000, type=int)
  parser.add_argument('--save_step', default=10000, type=int)
  parser.add_argument('--minibatch_size', default=4, type=int)

  parser.add_argument('--home_dir', type=str, default='/fs/clip-realspeech/projects/deep-rhythm/trainingSets/')
  parser.add_argument('--metadata_dir', required=True, type=str)
  parser.add_argument('--label_dir', required=True, type=str)
  parser.add_argument('--noise_condition', default='clean', type=str)
  parser.add_argument('--SNR', default=0, type=float, help='SNR in dB scale. 0 means signal and noise are equally loud.')
  parser.add_argument('--model_name', required=True, type=str)

  config = parser.parse_args()
  config.save_dir = f'models/{config.model_name}_PhoneRec__lrv{config.lr_v}_\
lrsr{config.lr_sr}_lossFct{config.loss_fct}_numSTRF{config.num_strfs}_\
updateSR{config.update_sr}_seed{config.strf_seed}_init{config.strf_init_method}_input{config.input_type}_\
convFeats{config.conv_feats}_batchSize{config.minibatch_size}/'
  try:
    mkdir(config.save_dir)
  except FileExistsError:
    print(f"Save_dir {config.save_dir} already exists. Overwritting...")
  Bs, As = read_cochba_j()

  @jit
  def wav2aud_lin(x):
    '''Output size: 200 x 128'''    
    return wav2aud_j(x, 5, 8, -2, 0, As, Bs, fft=True, return_stage=5).T
  batch_wav2aud_lin = vmap(wav2aud_lin)

  # @jit
  # def wav2aud_log(x, log_spec, pos):
  #   '''Output size: 200 x 128'''
  #   out = wav2aud_j(x, 5, 8, -2, 0, As, Bs, fft=True, return_stage=5).T
  #   eps = 1e-10
  #   out = jnp.log(jnp.array(out)+eps)
  #   out += -jnp.log(eps)
  #   #out = (out-jnp.mean(out))/jnp.var(out)
  #   return out
  # batch_wav2aud_log = vmap(wav2aud_log)
  
  # def batch_melspec(x):
  #   '''Output size: 4 x 200 x 128'''
  #   s = []
  #   if log_spec==0: 
  #     raise NotImplementedError
  #   eps = 1e-10
  #   for i in range(len(x)):
  #     temp = librosa.feature.melspectrogram(
  #       y=np.array(x[i,:]), sr=16000, n_fft=512, hop_length=80)[:,:200].T
  #     temp = jnp.log(jnp.array(temp)+eps)
  #     temp += -jnp.log(eps)
  #     #temp = (temp-jnp.mean(temp))/jnp.var(temp)
  #     s.append(temp)
  #   return jnp.stack(s)

  # def optax_ctc_loss(y, yh, y_pad):
  #   '''
  #   yh: 200 (time) * 77 (num_phone+1)
  #   y: 5 * 1 (not one-hot coded, labels start from 1)
  #   '''
  #   yh_pad = jnp.zeros((1, len(yh)))
  #   return optax.ctc_loss(jnp.expand_dims(yh, axis=0), yh_pad, jnp.expand_dims(y, axis=0), y_pad)
  # batch_ctc_loss = vmap(optax_ctc_loss)

  def xe_loss(y, yh, y_pad):
    '''y_pad is passed in as dummy argument for compatibility'''
    return optax.softmax_cross_entropy_with_integer_labels(yh, y)
  batch_xe_loss = vmap(xe_loss)

  @jit
  def forward_loss_batch(nn_params, s, sr, y, ypad):
    '''shape of input: batch is in the first (0) dimension'''
    yh = model.apply(nn_params, s, sr)
    return jnp.mean(loss_fct(y, yh, ypad))
  
  def update_batch_sr(nn_params, params, ov, osr, oState_nn, oState_diff, x, y, ypad): 
    if config.input_type=='audio':
      s = x.copy()
    elif config.spec_type=='logAud':
      s = batch_wav2aud_log(x)
    elif config.spec_type=='linAud':
      s = batch_wav2aud_lin(x)
    elif config.spec_type=='logMel':
      s = batch_melspec(x)
    else: raise KeyError
    loss, (g_v, g_sr) = value_and_grad(forward_loss_batch, (0, 2))(nn_params, s, params, y, ypad)
    u_v, oState_nn = ov.update(g_v, oState_nn)
    if config.update_sr==1:
      u_sr, oState_diff = osr.update(g_sr, oState_diff)
      return optax.apply_updates(nn_params, u_v), optax.apply_updates(params, u_sr), oState_nn, oState_diff, loss, g_sr
    elif config.update_sr==0:
      return optax.apply_updates(nn_params, u_v), params, oState_nn, oState_diff, loss, g_sr
      
  model = vSupervisedSTRF(n_phones = config.n_phones, 
                          input_type = config.input_type,
                          update_lin = config.update_lin,
                          use_class = config.use_class,
                          encoder_type = config.encoder_type, 
                          decoder_type = config.decoder_type, 
                          compression_method = config.compression_method,
                          conv_feats = config.conv_feats, 
                          pooling_stride = config.pooling_stride)
  
  params = {'sr': initialize_sr(config.num_strfs, config.strf_seed, 
                                method=config.strf_init_method)}
  if config.input_type == 'audio':
    params['compression_params'] = initialize_compression_params(val=1.0)
  if config.update_lin == True:
    params['alpha'] = jnp.array(0.9922179)
  
  if config.input_type == 'audio':
    nn_params = model.init(jax.random.key(0), 
                           jnp.ones([config.minibatch_size, 16000]), params)
  elif config.input_type == 'spec':
    nn_params = model.init(jax.random.key(0), 
                           jnp.ones([config.minibatch_size, 200, 128]), params)
  else: raise KeyError

  if config.loss_fct == 'ctc':
    loss_fct = batch_ctc_loss
    collapse_alignment = 1
  elif config.loss_fct == 'xe':
    loss_fct = batch_xe_loss
    collapse_alignment = 0
  train_set = SupervisedAudioDataset(
    home_dir=config.home_dir, metadata_dir=config.metadata_dir,
    label_dir=config.label_dir, collapse_alignment=collapse_alignment,
    noise_condition=config.noise_condition, snr=config.SNR)
  sampler = data.RandomSampler(train_set, replacement=True, 
                               num_samples=config.minibatch_size*config.num_steps)
  train_data_loader = data.DataLoader(train_set, batch_size=config.minibatch_size, 
                                      sampler=sampler)
  # TODO: add validation set
  
  optimizer_nn = optax.adam(config.lr_v)
  oState_nn = optimizer_nn.init(nn_params)
  
  optimizer_diff = optax.adam(config.lr_sr)
  oState_diff = optimizer_diff.init(params)
  
  print(f"The training set has {len(train_set)} utterances.")
  print(f'Number of STRFs: {config.num_strfs}')
  t0 = time.time()
  total_loss = 0
  step = 1
  with open(config.save_dir+'init.p', 'wb') as temp:
    pickle.dump([nn_params, params], temp)
  for (x,y,ypad) in tqdm(train_data_loader):
    x, y, ypad = jnp.array(x), jnp.array(y), jnp.array(ypad)
    nn_params, params, oState_nn, oState_diff, l, gradients_diff = update_batch_sr(
      nn_params, params, optimizer_nn, optimizer_diff, oState_nn, oState_diff, x, y, ypad)
    total_loss += l
    
    if step % config.save_step == 0:
      with open(config.save_dir+'training.log', 'a') as temp:
        temp.write(f"Step: {step}; loss: {total_loss/config.save_step}; Time: {(time.time()-t0)/60} min.\n")
      with open(config.save_dir+f'chkStep_{step}.p', 'wb') as temp:
        pickle.dump([nn_params, params], temp)
      t0 = time.time()
      total_loss = 0
    step += 1

class SupervisedAudioDataset(data.Dataset):
  '''
  A dataset to load audio and phone labels.
  '''
  def __init__(self, home_dir, metadata_dir, label_dir, collapse_alignment, noise_condition, snr,
               sampling=1., shuffle_utts=False):
    '''
    home_dir: str; path for AudSpec
    partition: list of tuples; home_dir/partition[i] should contains all .p files
        order: (noise, clean)
    sampling: the portion of entire data to take
    '''
    self.metadata_dir = join(home_dir, metadata_dir)
    self.label_dir = join(home_dir, label_dir)
    self.collapse_alignment = collapse_alignment
    self.noise_condition = noise_condition
    self.snr = snr

    with open(self.metadata_dir, 'r') as f:
      self.clean_utts = [line.strip('\n') for line in f.readlines()]
    with open(self.label_dir, 'r') as f:
      self.labels = [line.strip('\n') for line in f.readlines()]
  
    if sampling != 1.: # Subsample
      self.clean_utts = self.clean_utts[:int(len(self.clean_utts)*sampling)]
      self.labels = self.labels[:int(len(self.labels)*sampling)]
      self.clean_utt2len = {u: self.clean_utt2len[u] for u in self.clean_utts}

  def __len__(self):
    return len(self.clean_utts)
      
  def __getitem__(self, index_c):
    y = np.array(torch.load(self.labels[index_c]))
    # first_nonsil = np.argmax(y!=1).item()
    # if first_nonsil >= len(y)-210:
    #   label_start = np.random.randint(0,len(y)-210)
    # else:
    #   label_start = np.random.randint(first_nonsil, len(y)-210)
    label_start = np.random.randint(0,len(y)-215)
    y = y[label_start:(label_start+200)]
    
    xc, _ = librosa.load(path=self.clean_utts[index_c], sr=16000, offset=label_start*0.005, duration=1.0)
    xc = xc/np.sqrt(np.mean(xc**2)) # RMS normalization
    #print(self.labels[index_c], label_start, len(xc))

    if self.noise_condition == 'clean':
      xn = np.zeros(len(xc))
    elif self.noise_condition == 'white':
      xn = np.random.rand(len(xc))
      xn = xn/np.sqrt(np.mean(xn**2))
    elif self.noise_condition == 'pink':
      xn = colorednoise.powerlaw_psd_gaussian(1, len(xc))
      xn = xn/np.sqrt(np.mean(xn**2))

    SNR_linear = 10**(self.snr/10)
    xc = xc*np.sqrt(SNR_linear) + xn
    xc = xc/np.sqrt(np.mean(xc**2)) # RMS normalization   

    if self.collapse_alignment==1:
      y_out = [y[0]]
      for ch in y[1:]:
        if y_out[-1]!=ch:
          y_out.append(ch)
      # zero padding to keep output size constant
      y_out = np.pad(np.array(y_out), pad_width=(0,len(y)-len(y_out)), mode='constant')
  
      y_pad = np.zeros((1, len(y)))
      first_padded_index = np.where(y_out==0)[0][0].item()
      y_pad[first_padded_index:] = 1
      return xc, y_out, y_pad
    else:
      return xc, y, np.zeros(1)

if __name__ == '__main__':
  main() 