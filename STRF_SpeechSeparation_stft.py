import argparse
import glob
import time
import librosa
import pickle
from os import mkdir
from os.path import join
from tqdm import tqdm

import numpy as np
from math import floor, ceil
from torch.utils import data
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad, vmap, config 
from functools import partial
config.update("jax_enable_x64", True)

import optax
import flax.linen as nn
from random import random as random_random
from random import random, Random, sample

from strfpy import *
from strfpy_jax import *
from model.frontend import *
from model.loss import *
print(f"Training on {jax.default_backend()}...")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num_strfs', required=True, type=int) 
  parser.add_argument('--strf_seed', default=0, type=int) 
  parser.add_argument('--update_sr', default=1, type=int) 
  parser.add_argument('--input_type', default='spec', type=str, help='2 options: spec or audio')
  parser.add_argument('--conv_features', required=True, nargs="+", type=int)
  parser.add_argument('--encoder_type', default='strf', type=str)
  parser.add_argument('--strf_init_method', default='random', type=str, help='random or log. log currently only supports nm_strfs=40') 

  parser.add_argument('--lr_v', default=0.001, type=float)
  parser.add_argument('--lr_sr', default=0.001, type=float)
  parser.add_argument('--num_steps', default=200000, type=int)
  parser.add_argument('--save_step', default=10000, type=int)
  parser.add_argument('--minibatch_size', default=4, type=int)
  parser.add_argument('--loss', required=True, type=str, help='L1 or L2')

  parser.add_argument('--home_dir', default='/fs/clip-realspeech/projects/deep-rhythm/trainingSets/', type=str)
  parser.add_argument('--clean_dir', required=True, type=str)
  parser.add_argument('--noise_dir', required=True, type=str)
  parser.add_argument('--snr', required=True, type=float, help='SNR in dB scale. 0 means the signal has as much power as the noise')
  parser.add_argument('--model_name', required=True, type=str)

  config = parser.parse_args()
  config.save_dir = f'models/{config.model_name}_SpeechSep__SNR{config.snr}_\
encoderType{config.encoder_type}_numSTRF{config.num_strfs}_updateSR{config.update_sr}_\
init{config.strf_init_method}_input{config.input_type}_seed{config.strf_seed}_\
loss{config.loss}_convFeats{config.conv_features}_batchSize{config.minibatch_size}/'
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

  @jit
  def wav2aud_log(x, log_spec, pos):
    '''Output size: 200 x 128'''
    out = wav2aud_j(x, 5, 8, -2, 0, As, Bs, fft=True, return_stage=5).T
    eps = 1e-10
    out = jnp.log(jnp.array(out)+eps)
    out += -jnp.log(eps)
    #out = (out-jnp.mean(out))/jnp.var(out)
    return out
  batch_wav2aud_log = vmap(wav2aud_log)
  
  def batch_melspec(x):
    '''Output size: 4 x 200 x 128'''
    s = []
    if log_spec==0:
      raise NotImplementedError
    eps = 1e-10
    for i in range(len(x)):
      temp = librosa.feature.melspectrogram(y=np.array(x[i,:]), sr=16000, n_fft=512, hop_length=80)[:,:200].T
      temp = jnp.log(jnp.array(temp)+eps)
      temp += -jnp.log(eps)
      #temp = (temp-jnp.mean(temp))/jnp.var(temp)
      s.append(temp)
    return jnp.stack(s)

  def audspec_loss(s, s_hat, loss=config.loss):
    if loss == 'L2':
      return jnp.mean((s-s_hat)**2)
    elif loss == 'L1':
      return jnp.mean(jnp.abs(s-s_hat))
    else: 
        raise NotImplementedError
  batch_audspec_loss = vmap(audspec_loss)

  @jit
  def forward_loss_batch(variables, xn, xc, sr):
    '''shape of input: batch is in the first (0) dimension'''
    s_hat = model.apply(variables, xn, sr)
    #print('Before entering loss function:', xc.shape, s_hat.shape)
    return jnp.mean(vbsrnn_loss(xc, s_hat))
  
  def update_batch_sr(variables, sr, ov, osr, opt_state_v, opt_state_sr, xn, xc): 
    loss, (g_v, g_sr) = value_and_grad(forward_loss_batch, (0, 3))(variables, xn, xc, sr)
    u_v, opt_state_v = ov.update(g_v, opt_state_v)

    if config.update_sr==1:
      u_sr, opt_state_sr = osr.update(g_sr, opt_state_sr)
      return optax.apply_updates(variables, u_v), optax.apply_updates(sr, u_sr), opt_state_v, opt_state_sr, loss
    elif config.update_sr==0:
      return optax.apply_updates(variables, u_v), sr, opt_state_v, opt_state_sr, loss

  model = vAudioSTRFAE(input_type = config.input_type,
                       encoder_type = config.encoder_type, 
                       conv_features = config.conv_features)
  
  sr = initialize_sr(config.num_strfs, config.strf_seed, method=config.strf_init_method)
  variables = model.init(jax.random.key(config.strf_seed), jnp.ones([config.minibatch_size, 16000]), sr)

  train_set = CocktailAudioDataset(home_dir=config.home_dir, clean_dir=config.clean_dir, noise_dir=config.noise_dir, 
                                   sampling=1., snr=config.snr)
  sampler = data.RandomSampler(train_set, replacement=True, num_samples=config.minibatch_size*config.num_steps)
  train_data_loader = data.DataLoader(train_set, batch_size=config.minibatch_size, sampler=sampler)
  # TODO: add validation set
  
  op_v, op_sr = optax.adam(config.lr_v), optax.adam(config.lr_sr)
  opt_state_v = op_v.init(variables)
  opt_state_sr = op_sr.init(sr)
  print(f"The training set has {len(train_set)} utterances.")
  print(f'Number of STRFs: {config.num_strfs}')
  t0 = time.time()
  total_loss = 0
  step = 1
  with open(config.save_dir+'init.p', 'wb') as temp:
    pickle.dump([variables, sr], temp)
  for xn, xc in tqdm(train_data_loader):
    xn, xc = jnp.array(xn), jnp.array(xc)
    variables, sr, opt_state_v, opt_state_sr, l = update_batch_sr(
      variables, sr, op_v, op_sr, opt_state_v, opt_state_sr, xn, xc)
    total_loss += l
    
    if step % config.save_step == 0:
      with open(config.save_dir+'training.log', 'a') as temp:
        temp.write(f"Step: {step}; loss: {total_loss/config.save_step}; Time: {(time.time()-t0)/60} min.\n")
      with open(config.save_dir+f'chkStep_{step}.p', 'wb') as temp:
        pickle.dump([variables, sr], temp)
      t0 = time.time()
      total_loss = 0
    step += 1

@jit 
def stft_wav2spec(xwav, n_fft=256, hop_length=80):
  _, _, x = jax.scipy.signal.stft(xwav, nperseg = n_fft, noverlap = n_fft-hop_length)
  return x
#batch_stft = vmap(stft, in_axes=(0, None, None))

def istft_spec2wav(spec, n_fft=256, hop_length=80):
  _, x = jax.scipy.signal.istft(spec, nperseg = n_fft, noverlap = n_fft-hop_length)
  return x
#batch_istft = vmap(istft, in_axes=(0, None, None))

class audioSTRFAE(nn.Module):
  """Convolutional Decoder."""
  encoder_type: str
  conv_features: int
  input_type: str
  
  def setup(self):
    if self.input_type == 'audio':
      self.audspec = AuditorySpectrogram(input_length=16000)
    elif self.input_type == 'spec':
      with np.load('cochlear_filter_params.npz') as data:
        self.Bs, self.As = jnp.array(data['Bs']), jnp.array(data['As'])
    else: raise KeyError

  def __call__(self, x, sr):
    x_spec = stft_wav2spec(x)[:,:200]
    y = self.conv(self.encode(x, sr))
    #print("When applying mask:", x_spec.shape, y.shape)
    y = istft_spec2wav(x_spec * y)
    y = jnp.pad(y, (0, 80))
    return y

  def encode(self, x, sr):
    '''Use STRFs to encode stimuli'''
    # Audio to spectrogram
    if self.input_type == 'audio':
      x = self.audspec(x)[:-1, :].T
    elif self.input_type == 'spec':
      x = wav2aud_j(x, 5, 8, -2, 0, self.As, self.Bs).T
    #print('After wav2aud', x.shape)

    if self.encoder_type=='strf':
      x = strf(x, sr).real.transpose(2, 1, 0)
    elif self.encoder_type=='cnn':
      x = jnp.expand_dims(x.T, axis=2)
    else: raise KeyError

    return x

  @nn.compact
  def conv(self, x):
    '''Input: frequency (128) x time (200/s) x n_channel (e.g. 198)'''
    for i in range(len(self.conv_features)):
      x = nn.Conv(features=self.conv_features[i], kernel_size=(3, 3), 
                  strides=(1,1))(x) 
      x = nn.gelu(x)
      # if i == len(self.conv_features)-1:
      #   x = nn.sigmoid(x)
      # else:
      #   x = nn.gelu(x)
    x = x.squeeze().T
    #print('Befor last dense layer', x.shape)
    x = nn.Dense(features=129)(x).T
    x = nn.sigmoid(x)
    return x

vAudioSTRFAE = nn.vmap(
    audioSTRFAE,
    in_axes=(0, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    methods=["__call__", "conv", "encode"])

class CocktailAudioDataset(data.Dataset):
  def __init__(self, home_dir, clean_dir, noise_dir, sampling=1., snr=1., shuffle_utts=False):
    '''
    home_dir: str; path for AudSpec
    partition: list of tuples; home_dir/partition[i] should contains all .p files
        order: (noise, clean)
    sampling: the portion of entire data to take
    '''
    self.snr = snr
    self.clean_dir = join(home_dir, clean_dir)
    self.noise_dir = join(home_dir, noise_dir)

    with open(self.clean_dir, 'r') as f:
      self.clean_utts = [line.strip('\n') for line in f.readlines()]
    with open(self.noise_dir, 'r') as f:
      self.noise_utts = [line.strip('\n') for line in f.readlines()]

    self.clean_utt2len = {utt:librosa.get_duration(path=utt) for utt in self.clean_utts}
    self.noise_utt2len = {utt:librosa.get_duration(path=utt) for utt in self.noise_utts}
  
    if sampling != 1.: # Subsample
      self.clean_utts = self.clean_utts[:int(len(self.clean_utts)*sampling)]
      self.clean_utt2len = {u: self.clean_utt2len[u] for u in self.clean_utts}
      raise NotImplementedError
    
  def rms(self, x):
    norm = np.sqrt(np.mean(x**2))
    if norm != 0:
      x /= norm
    return x

  def __len__(self):
    return len(self.clean_utts)
      
  def __getitem__(self, index_c):
    clean_utt = self.clean_utts[index_c]
    x_len = self.clean_utt2len[clean_utt]
    start = random_random()*(x_len-1)
    xc, _ = librosa.load(path=clean_utt, sr=16000, offset=start, duration=1.0) 
    xc = self.rms(xc) # RMS normalization
  
    noise_utt = sample(self.noise_utts, 1)[0]
    x_len = self.noise_utt2len[noise_utt]
    start = random_random()*(x_len-1)
    xn, _ = librosa.load(path=noise_utt, sr=16000, offset=start, duration=1.0) 
    xn = self.rms(xn) # RMS normalization

    SNR_linear = 10**(self.snr/10)
    xm = xc*np.sqrt(SNR_linear) + xn
    xm = self.rms(xm) # RMS normalization
    return xm, xc

if __name__ == '__main__':
  main() 