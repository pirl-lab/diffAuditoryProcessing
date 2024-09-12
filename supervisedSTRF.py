import numpy as np
from jax import numpy as jnp 
from jax import vmap

import flax.linen as nn
from model.frontend import *
from strfpy_jax import *


class supervisedSTRF(nn.Module):
  '''
  A STRF model that takes in a time-frequency representation (e.g. 
  Auditory Spectrogram) and outputs a series of phone labels, 
  time-aligned with the time-frequency representation. 

  It uses a STRF layer followed by a CNN. 
  '''
  n_phones: int
  input_type: str
  
  encoder_type: str
  decoder_type: str
  conv_feats: int
  compression_method: str
  update_lin: bool
  pooling_stride: int = 3
  use_class: bool = False

  def setup(self):
    if self.input_type=='audio':
      if self.use_class:
        self.audspec = AuditorySpectrogram(input_length=16000)
      else:
        with np.load('cochlear_filter_params.npz') as data:
          self.Bs, self.As = jnp.array(data['Bs']), jnp.array(data['As'])
        self.LIN = nn.Conv(features=1, kernel_size=(2,), strides=(1,))


  def wav2aud(self, x, fac, alpha):
    '''
    Takes waveform in and output the auditory spectrogram in a differentiable
    manner. 

    Output size: 200 x 128
    '''
    if self.update_lin == False:
      x = wav2aud_j(x, 5, 8, fac, 0, As=self.As, Bs=self.Bs, 
                      compression_method=self.compression_method).T
    else:
      x = wav2aud_j(x, 5, 8, fac, 0, As=self.As, Bs=self.Bs, 
                    return_stage=1)
      x = compression(x, fac, method=self.compression_method)
      x = jnp.expand_dims(x, axis=-1)
      x = self.LIN(x)
      x = x.squeeze()
      x = nn.relu(x)

      # leaky integration here
      out = []
      #print(x.shape)
      for i in range(len(x)):
          out.append(leaky_integrator_fft(x[i,:], alpha)) # This can be vectorized
      x = jnp.vstack(out)
      x = x.real
      L_frm = 80
      x = x[:, (L_frm - 1)::L_frm].T
    
    return x
  
  # def __call__(self, x, params):
  #   return self.wav2aud(x, params['compression_params'], params['alpha'])

  def __call__(self, x, aud_params):
    return self.conv(self.encode(x, aud_params))

  def encode(self, x, params):
    '''
    Use STRFs to encode stimuli.
    
    Params: a dictionary. The following values are accepted:
    'compression_params': comparession parameters. one-dim vector whose length 
                          equals n_channel+1 (due to first difference).
    'sr': scale and rates, STRF parameters. 

    TODO: make frmlen, time constant, and octave_shift specified outside 
    this function
    '''
    
    # Cochlear step; skipped if input is spectrogram already
    if self.input_type == 'audio':
      if self.use_class:
        out = self.audspec(x).T
      elif 'alpha' not in params.keys():
        out = self.wav2aud(x, params['compression_params'], alpha=0.9922179)
      else:
        out = self.wav2aud(x, params['compression_params'], 
                            alpha=params['alpha'])

    elif self.input_type == 'spec':
      out = x.copy()
    else: raise KeyError
    #print(x.shape, out.shape)

    # Cortical step; STRF or CNN
    if self.encoder_type=='strf':
      out = strf(out, params['sr']).real
      out = out.transpose(2, 1, 0)
    elif self.encoder_type=='cnn':
      out = jnp.expand_dims(out.T, axis=2)
    else: raise NotImplementedError
    
    return out

  @nn.compact
  def conv(self, x):
    '''Input: frequency (128) x time (200/s) x n_channel (e.g. 30)'''
    
    if self.decoder_type == 'mlp':
      raise NotImplementedError
    elif self.decoder_type == 'cnn':
      for i in range(len(self.conv_feats)):
        x = nn.Conv(features=self.conv_feats[i], kernel_size=(3, 3))(x) 
        x = nn.gelu(x)
        x = nn.avg_pool(x, window_shape=(self.pooling_stride,1), 
                        strides=(self.pooling_stride,1))

      x = x.transpose(1, 0, 2)
      #x = x.reshape(len(x), -1)
      x = nn.Dense(features=self.n_phones+1)(x)
      #x = nn.gelu(x)
    # output: 200 (time) x 76 (n_phones)
    #x = x.T # 0218 - output: 76 (n_phones) x 200 (time)
    return x

vSupervisedSTRF = nn.vmap(
    supervisedSTRF,
    in_axes=(0, None), out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    methods=["__call__", "conv", "encode"])


if __name__ == "__main__":
  model = vSupervisedSTRF(n_phones = 42, 
                          input_type = 'spec',
                          compression_method = 'power',
                          encoder_type = 'strf', 
                          decoder_type = 'cnn', 
                          conv_feats = [10,20,40], 
                          pooling_stride = 3)

  params = {'sr': initialize_sr(40, 1),
            'compression_params': initialize_compression_params()}

  variables = model.init(jax.random.key(0), 
                         jnp.ones([4, 200, 128]), params)