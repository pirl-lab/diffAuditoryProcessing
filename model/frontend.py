import numpy as np 
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import flax.linen as nn
from jax.scipy.signal import stft, istft

import os


class CochlearFilter_Roex_fft(nn.Module):
  '''
  The filterbank used in Auditory Spectrogram (chi2005). 

  Applies filtering through fft -> multiplying.
  '''
  input_length: int # Used to calculate filter duration
  sr: int = 16000  # Default sample rate
  
  def setup(self):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with np.load(os.path.join(current_dir, 'cochlear_filter_params.npz')) as data:
      Bs, As = jnp.array(data['Bs']), jnp.array(data['As'])

    freqs = jnp.fft.fftfreq(self.input_length)*2*np.pi
    e_jw = jnp.cos(freqs) - 1j*jnp.sin(freqs)

    self.Hs = jnp.zeros([As.shape[0], self.input_length], dtype=jnp.complex128)
    for i in range(As.shape[0]):
      b, a = Bs[i,:], As[i,:]
      H = jnp.sum(jnp.array([b[i]*e_jw**i for i in range(25)]), axis=0) 
      H /= jnp.sum(jnp.array([a[i]*e_jw**i for i in range(25)]), axis=0)
      self.Hs = self.Hs.at[i, :].set(H)
    
  def __call__(self, x):
    X = jnp.fft.fft(x)
    cochleagram = []
    for i in range(self.Hs.shape[0]):
      cochleagram.append(jnp.fft.ifft(self.Hs[i,:]*X).real)
    cochleagram = jnp.stack(cochleagram)
    #cochleagram = cochleagram.transpose(1,0,2)
    return cochleagram
  
  def invert(self, x):
    '''
    Takes in a cochleagram and invert it.
    x: 129 x n_time_samples
    '''
    out = []
    for i in range(self.Hs.shape[0]):
      X = jnp.fft.fft(x[i,:])
      out.append(jnp.fft.ifft(X/self.Hs[i,:]).real)
    #print(out.shape)
    return jnp.mean(jnp.stack(out), axis=0).real
  
vCochlearFilter_Roex_fft = nn.vmap(
  CochlearFilter_Roex_fft,
  in_axes=(0,),  # Input x has shape (batch_size, input_length)
  out_axes=0,  # The output should also have a batch dimension
  variable_axes={'params': None},  # Do not vectorize over parameters
  split_rngs={'params': False},  # Do not split random number generators
  methods=["__call__", "invert"]  
)
 
class PowerLawCompression(nn.Module):
  '''
  Applies power law compression to each channel: 
  y = x^a

  a is a different value per frequency channel, and is learnable through back-
  propagation. 

  Input x: n_channels x n_timeSamples
  new input x: B x H x W
  '''
  sr: int = 16000  # Default sample rate
  input_channels: int = 129 # Default number of freq channels
  init_value: float = 1.0

  def setup(self):
    self.alpha = self.variable("params", "alpha", lambda: jnp.ones(self.input_channels)*0.9)

  def __call__(self, x):
    if len(x.shape) == 2: # H x W
      for i in range(x.shape[0]):
        x = x.at[i,:].set(
          jnp.abs(x[i,:]) ** self.alpha.value[i] * jnp.sign(x[i,:])
        )


      #x = self.compression(x.T).T
    else: # B x H x W
      # x = jnp.transpose(x, (0, 2, 1)) # B x W x H
      # x = self.compression(x)
      # x = jnp.transpose(x, (0, 2, 1)) # B x H x W
      raise NotImplementedError
    return x
  
class LateralInhibitionNetwork(nn.Module):
  '''
  Applies LIN, which is composed of linear convolution (along freq.) and ReLU.
  '''
  def setup(self):
    def initializer_first_diff(key, shape, dtype=jnp.float64):
      kernel = jnp.array([1, -1], dtype=jnp.float64)
      return kernel.reshape(shape)
    self.LI_conv = nn.Conv(features=1, kernel_size=(2,), strides=(1,),
                           kernel_init=initializer_first_diff,
                           dtype=jnp.float64, use_bias=False)


  def __call__(self, x):
    '''
    Input x: H x W
    new Input x: B x H x W
    '''
    if len(x.shape) == 2:
      x = jnp.expand_dims(x.T, axis=-1)
      x = self.LI_conv(x)
      x = x.squeeze().T
    else:
      x = jnp.transpose(x, (0, 2, 1)) # B x W x H
      x = jnp.expand_dims(x, axis=-1) # B x W x H x 1
      x = self.LI_conv(x)
      x = jnp.squeeze(x, axis=-1)
      x = jnp.transpose(x, (0, 2, 1)) # B x H x W
    x = nn.relu(x)
    return x
  
class LeakyIntegration(nn.Module):
  '''
  Applies leaky integration followed by downsampling.
  Leaky integrator was implemented by filtering in the FFT domain.
  This version is not vectorized. Use the vmap version for batch handling.

  alpha is a learable parameter. 
  '''
  input_length: int
  downsample_rate: int = 80

  def setup(self):
    self.alpha = self.variable("params", "alpha", lambda: jnp.array(0.9922179))
    self.freqs = jnp.fft.fftfreq(self.input_length) * 2 * jnp.pi

  @nn.compact
  def __call__(self, x):
    out = []
    for i in range(x.shape[0]):
      X = jnp.fft.fft(x[i,:])
      H = 1 / (1 - self.alpha.value * 
               (jnp.cos(self.freqs) - 1j*jnp.sin(self.freqs))
               )
      out.append(jnp.fft.ifft(H * X).real)
    out = jnp.stack(out)

    # Downsample
    
    out = out[:, (self.downsample_rate - 1)::self.downsample_rate]
    return out
  
vLeakyIntegration = nn.vmap(
LeakyIntegration,
in_axes=(0,),  # Assuming the input x has shape (batch_size, input_length)
out_axes=0,  # The output cochleagram should also have a batch dimension
variable_axes={'params': None},  # Do not vectorize over parameters
split_rngs={'params': False},  # Do not split random number generators
methods=["__call__"]  # Vectorize the __call__ method
)

class AuditorySpectrogram(nn.Module):
  '''
  Auditory Spectrogram (chi2005). Five steps:

  1. Filterbank (roex)
  2. Compression (logistic or power law)
  3. First difference; 4. ReLU (3 & 4 in one step called `LIN`)
  5. Downsampling
  '''
  input_length: int # Used to calculate filter duration
  sr: int = 16000  # Default sample rate
  frame_length: int = 5 # duration of a frame
  
  def setup(self):
    self.downsample_rate = int(self.sr / (1000/self.frame_length))

    self.cochlearFilterbank = CochlearFilter_Roex_fft(
      input_length = self.input_length, 
      sr = self.sr)
    self.compression = PowerLawCompression(sr = self.sr)
    self.LIN = LateralInhibitionNetwork()
    self.LeakyIntegration = LeakyIntegration(
      input_length = self.input_length,
      downsample_rate = self.downsample_rate)

    
  def __call__(self, x):
    #print(x.shape)
    x = self.cochlearFilterbank(x)
    #print(x.shape)
    x = self.compression(x)
    x = self.LIN(x)
    x = self.LeakyIntegration(x)
    return x
  
class vAuditorySpectrogram(nn.Module):
  '''
  Auditory Spectrogram (chi2005). Five steps:

  1. Filterbank (roex)
  2. Compression (logistic or power law)
  3. First difference; 4. ReLU (3 & 4 in one step)
  5. Downsampling
  '''
  input_length: int # Used to calculate filter duration
  sr: int = 16000  # Default sample rate
  frame_length: int = 5 # duration of a frame
  
  def setup(self):
    self.downsample_rate = int(self.sr / (1000/self.frame_length))

    self.cochlearFilterbank = vCochlearFilter_Roex_fft(
      input_length = self.input_length, 
      sr = self.sr)
    self.compression = PowerLawCompression(sr = self.sr)
    self.LIN = LateralInhibitionNetwork()
    self.LeakyIntegration = vLeakyIntegration(
      input_length = self.input_length,
      downsample_rate = self.downsample_rate)

    
  def __call__(self, x):
    x = self.cochlearFilterbank(x)
    x = self.compression(x)
    x = self.LIN(x)
    x = self.LeakyIntegration(x)
    return x
  
class STFTSpectrogram():
  '''
  Dummy for Mel Spectrogram
  '''
  sr: int = 16000  # Default sample rate
  n_fft: int = 512
  frame_length: float = 5.0
  
  def __init__(self):
    self.hop_length = int(self.sr/1000*self.frame_length)
    
  def stft(self, x):
    _, _, x = stft(x, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
    return x
  
  def istft(self, x):
    _, x = istft(x, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
    return x
