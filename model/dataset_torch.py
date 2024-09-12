import librosa
from os.path import join
from torch.utils import data

from random import random, Random, sample
import numpy as np

class CocktailAudioDataset_torch(data.Dataset):
  def __init__(self, home_dir, clean_dir, noise_dir, duration=1.0,
               sampling=1., snr=1., shuffle_utts=False):
    '''
    home_dir: str; path for AudSpec
    partition: list of tuples; home_dir/partition[i] should contains all .p files
        order: (noise, clean)
    sampling: the portion of entire data to take
    '''
    self.snr = snr
    self.clean_dir = join(home_dir, clean_dir)
    self.noise_dir = join(home_dir, noise_dir)
    self.duration = duration

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

  def __len__(self):
    return len(self.clean_utts)
  
  def rms(self, x):
    norm = np.sqrt(np.mean(x**2))
    if norm != 0:
      x /= norm
    return x
      
  def __getitem__(self, index_c):
    clean_utt = self.clean_utts[index_c]
    x_len = self.clean_utt2len[clean_utt]
    start = random()*(x_len-1)
    xc, _ = librosa.load(path=clean_utt, sr=16000, offset=start, 
                         duration=self.duration) 
    xc = self.rms(xc) # RMS normalization
  
    noise_utt = sample(self.noise_utts, 1)[0]
    x_len = self.noise_utt2len[noise_utt]
    start = random()*(x_len-1)
    xn, _ = librosa.load(path=noise_utt, sr=16000, offset=start, 
                         duration=self.duration) 
    xn = self.rms(xn)

    SNR_linear = 10**(self.snr/10)
    xm = xc*np.sqrt(SNR_linear) + xn
    xm = self.rms(xm) # RMS normalization
    return xm, xc
  

