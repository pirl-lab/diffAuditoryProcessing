import librosa
from tqdm import tqdm
import pickle
import glob
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax.scipy.signal import convolve as jconvolve 
from jax import config
config.update("jax_enable_x64", True)

from functools import partial

from math import ceil,floor
from scipy.signal import lfilter
from strfpy import *

def read_cochba_j():
    with open('./cochba.txt') as f:
        file = f.readlines()
    cochba = []
    for line in file:
        line = line.strip("\n").split("\t")
        if line == "": continue
        cochba.append([])
        for num in line:
            num = num.replace("i","j")
            num = num.replace(" ","")
            cochba[-1].append(complex(num))
    cochba = jnp.array(cochba)
    Bs, As = [], []
    ps = [p.real for p in list(cochba[0, :])] # beware of 0-indexing!
    for ch, p in enumerate(ps):
        temp = cochba[1:p.astype(int)+2,ch]
        B, A = jnp.array([f.real for f in temp]), jnp.array([f.imag for f in temp])
        B = jnp.pad(B, (0, 25-len(temp)), mode='constant')
        A = jnp.pad(A, (0, 25-len(temp)), mode='constant')
        Bs.append(B)
        As.append(A)  
    Bs = jnp.array(np.vstack(Bs))
    As = jnp.array(np.vstack(As))
    return Bs, As 
Bs, As = read_cochba_j()

def sigmoid_j(y, fac):
    '''
    Copied from MATLAB documentation: nonlinear function for cochlear model.
    
    fac: nonlinear factor.
    	 -- fac > 0, transister-like function
    	 -- fac = 0, hard-limiter
    	 -- fac = -1, half-wave rectifier
    	 -- else, no operation, i.e., linear 

    SIGMOID is a monotonic increasing function which simulates 
    hair cell nonlinearity. 
    '''
    if fac > 0: 
        y = jnp.exp(-y/fac)
        y = 1/(1+y)
    elif fac == 0:
        y = (y > 0) # Do I need to turn this into integer?
    elif fac == -3:
        raise NotImplementedError # halfregu()
    return y

@jit
def cochlear_filter_fft(b, a, x):
    '''
    Given cochlear filter coefficients b and a, apply the filter on signal x by 
    multiplying the transfer function in the fft domain (followed by ifft). 

    Modified from recursive_iir_jax for a and b strictly of duration 25.
    Assume that the initial condition is rest.
    
    x: length * dimensions. If the second dimension exists, the same filter 
    would be applied along each dimension.
    '''
    assert len(b) == 25
    assert len(a) == 25

    freqs = jnp.fft.fftfreq(len(x))*2*np.pi
    e_jw = jnp.cos(freqs) - 1j*jnp.sin(freqs)
    H = jnp.sum(jnp.array([b[i]*e_jw**i for i in range(25)]), axis=0) 
    H /= jnp.sum(jnp.array([a[i]*e_jw**i for i in range(25)]), axis=0)

    X = jnp.fft.fft(x)
    X = jnp.fft.ifft(H*X)
    #X = jnp.abs(X)
    return X

@jit
def cochlear_filter(b, a, x):
    '''
    Given cochlear filter coefficients b and a, apply the filter on signal x by 
    recursive computation using ARMA in the time domain. 
    When gradients are taken, this could be extremely slow due to how jax handles
    for-loops and the fact that for-loop is large in time domain. 

    Modified from recursive_iir_jax for a and b strictly of duration 25.
    Assume that the initial condition is rest.
    
    x: length * dimensions. If the second dimension exists, the same filter would be applied along each dimension.
    '''
    assert len(b) == 25
    assert len(a) == 25
    
    a1 = a[1:]
    ar = jconvolve(x, b) # AR part
    y = jnp.zeros(len(x)+24)
    
    for i in range(len(x)):
        t0 = time.time()
        if len(a) > 1:
            ma = jnp.sum(y[i:i+24] * a1[::-1])
        else:
            ma = 0
        y = y.at[i+24].set(ar[i] - ma)
    return y[24:]

@jit
def leaky_integrator_fft(x, alpha):
    ''' 
    Apply the leaky integrator by filtering in fft domain. 
    Would this be faster if it were a convolution? 
    b = [1], a = [1, -alpha] '''
    #alpha = 0.98
    X = jnp.fft.fft(x)
    freqs = jnp.fft.fftfreq(len(x))*2*np.pi
    H = 1 / (1 - alpha * (jnp.cos(freqs) - 1j*jnp.sin(freqs)))
    return jnp.fft.ifft(H*X)

    
def wav2aud_j(x, frmlen, time_constant, fac, octave_shift, 
              As, Bs, filt_type='p', compression_method='identity',
              fft=True, return_stage=5, downsample='default'):
    '''
    Leslie: implementation of matlab wav2aud2.m in Ding et al., 2017
    
    WAV2AUD computes the auditory spectrogram for an acoustic waveform.
    This function takes the advantage of IIR filter's fast performance
    which not only reduces the computaion but also saves remarkable
    memory space.
    
    x: wavform input
    frmlen: frame length in ms, typical 8, 16, exponent of 2
    time_constant: time constant in ms, typically 4, 16, 64 etc.
    fac: nonlinear factor (critical level ratio), see sigmoid()
    octave_shift: shifted by # of octave, e.g. 0 for 16k, -1 for 8k.
    sf = 16k * 2^[octave_shift]
    filt_type: filter type. Currently only implemented 'p', Powen's IIR filter
    '''    
    assert As.shape == Bs.shape
    
    if filt_type != 'p': raise NotImplementedError
    L_frm = round(frmlen * 2**(4+octave_shift)) # frame length (points)
    N = ceil(len(x)/L_frm) # number of frames
    x = jnp.pad(x, (0, N*L_frm-len(x)), mode='constant') # zero-padding the signal
    
    if time_constant != 0: 
        alpha = jnp.exp( -1 / (time_constant * 2**(4+octave_shift)) )
    else:
        alpha = 0

    v5 = []
    for ch in range(len(As)):
        B, A = Bs[ch,:], As[ch,:]
        if fft:
            y = cochlear_filter_fft(B, A, x) # -> y1
        else:
            y = cochlear_filter(B, A, x) # -> y1
        #y = sigmoid_j(y, fac) # -> y2
        # Useless for now, since fac=-2, leading to linear operation
        v5.append(y)
    v5 = jnp.vstack(v5)
    if return_stage==1:return v5
    v5 = compression(v5, fac, method=compression_method)
    if return_stage==2: return v5
    
    # Apply a first difference filter -> y3
    v5 = v5[:-1, :] - v5[1:, :]
    if return_stage==3: return v5
    
    # Half wave rectifier -> y4
    v5 = v5.at[:,:].max(0)
    if return_stage==4: return v5
    
    if time_constant != 0: # leaky integration -> y5        
        out = []
        for i in range(len(v5)):
            out.append(leaky_integrator_fft(v5[i,:], alpha)) # This can be vectorized
        v5 = jnp.vstack(out)
        v5 = v5.real
        if downsample=='default':
            inds = jnp.arange(1, N+1)*L_frm-1
            v5 = v5[:,inds]
        else:
            v5 = v5[:, (L_frm - 1)::L_frm]
    elif L_frm == 1: 
        pass
    else:
        raise NotImplementedError
    if return_stage==5: return v5

def cochleagram2aud(v5, frmlen, time_constant, fac, octave_shift, 
                    compression_method='identity'):
    '''
    Leslie: implementation of matlab wav2aud2.m in Ding et al., 2017
    
    WAV2AUD computes the auditory spectrogram for an acoustic waveform.
    This function takes the advantage of IIR filter's fast performance
    which not only reduces the computaion but also saves remarkable
    memory space.
    
    x: wavform input
    frmlen: frame length in ms, typical 8, 16, exponent of 2
    time_constant: time constant in ms, typically 4, 16, 64 etc.
    fac: nonlinear factor (critical level ratio), see sigmoid()
    octave_shift: shifted by # of octave, e.g. 0 for 16k, -1 for 8k.
    sf = 16k * 2^[octave_shift]
    filt_type: filter type. Currently only implemented 'p', Powen's IIR filter
    '''    
    L_frm = round(frmlen * 2**(4+octave_shift)) # frame length (points)
    
    if time_constant != 0: 
        alpha = jnp.exp( -1 / (time_constant * 2**(4+octave_shift)) )
    else:
        alpha = 0

    v5 = compression(v5, fac, method=compression_method)
    
    # Apply a first difference filter -> y3
    #v5 = v5[:-1, :] - v5[1:, :]
    
    # Half wave rectifier -> y4
    v5 = v5.at[:,:].max(0)
    
    if time_constant != 0: # leaky integration -> y5        
        out = []
        for i in range(len(v5)):
            out.append(leaky_integrator_fft(v5[i,:], alpha)) # This can be vectorized
        v5 = jnp.vstack(out)
        v5 = v5.real
        # inds = jnp.arange(1, N+1)*L_frm-1
        # v5 = v5[:,inds]
        v5 = v5[:, (L_frm - 1)::L_frm]
    elif L_frm == 1: 
        pass
    else:
        raise NotImplementedError
    return v5

def root_norm(x, p):
    norm_constant = jnp.sum(x**p)/len(x)
    return (x/norm_constant)**(1/p)

def power_norm(x, alpha):
    return jnp.abs(x)**alpha * jnp.sign(x)

def compression(mag, fac, method='identity'):
    '''
    y: auditory spectrogram. n_channels x n_timebins
    '''
    # Insert dummy dimension of 1 if y and fac has only one channel
    if len(mag.shape) == 1:
        y = jnp.expand_dims(mag, axis=0)
    else:
        y = mag

    #print(y.shape)
    if method=='logistic':
        for i in range(y.shape[0]):
            y = y.at[i,:].set(1/(1+jnp.exp(y[i,:]/fac[i])))
    elif method == 'root':
        for i in range(y.shape[0]):
            # norm_factor = jnp.sum(y[i,:]**fac[i])**(1/fac[i])
            # y = y.at[i,:].set(y[i,:]/norm_factor)
            y = y.at[i,:].set(root_norm(y[i,:], p=fac[i]))
    elif method == 'power':
        for i in range(y.shape[0]):
            y = y.at[i,:].set(power_norm(y[i,:], alpha=fac[i]))
    elif method == 'identity':
        pass
    else:
        raise KeyError
    return y


def inverse_cochlear_filter(Bs, As, v):
    '''
    v: cochleagram of shape (n_channels, n_time_samples)
    '''
    xs = []
    for ch in range(len(As)):
        b, a = Bs[ch,:], As[ch,:]

        freqs = jnp.fft.fftfreq(v.shape[1])*2*np.pi
        e_jw = jnp.cos(freqs) - 1j*jnp.sin(freqs)
        H = jnp.sum(jnp.array([b[i]*e_jw**i for i in range(25)]), axis=0) 
        H /= jnp.sum(jnp.array([a[i]*e_jw**i for i in range(25)]), axis=0)
        V = jnp.fft.fft(v[ch, :])
        V = jnp.fft.ifft(V/H)
        xs.append(V)
    xs = jnp.vstack(xs)
    xs = jnp.mean(xs, axis=0)
    return xs.real

###################################
##### Cortical models (STRFs) #####

#@jit
def aud2cor_j(y, paras, rv, sv):
    '''
    Python version of aud2cor() in the NSL toolbox. From auditory spectrogram to cortical STRF.

    Input:
    y: auditory spectrogram
    paras: parameters that generated the auditory spectrogram, e.g. [5, 8, -2, 0]
    rv: rate vector
    sv: scale vector
    out_filename: write the output in this file
    disp: normalization during display
    '''
    
    if len(paras) < 5: 
        FULLT, BP = 0, 0
        FULLX = FULLT
    else:
        raise NotImplementedError
    
    K1, K2 = len(rv), len(sv)
    N, M = y.shape
    # Parsing paras
    STF = 1000/paras[0]
    if M == 95: SRF = 20
    else: SRF = 24 # Why??

    dM, dN = 0, 0
    N1, M1 = 2**ceil(np.log2(N)), 2**ceil(np.log2(M))
    N2, M2 = N1*2, M1*2
    
    Y = jnp.fft.fft(y, M2, axis=1)[:,:M1] # Fourier transform (frequency)
    Y = jnp.fft.fft(Y[:N,:], N2, axis=0) # Fourier transform (temporal)

    cr = jnp.zeros([K2, K1*2, N+2*dN, M+2*dM],dtype=complex)
    for rdx in range(K1): # rate filtering
        HR = gen_cort_j(rv[rdx], N1, STF, [rdx+1+BP, K1+BP*2])

        for sgn in [1, -1]:
            if sgn == 1: 
                HR = jnp.concatenate([HR, np.zeros(N1)])
            else: # conjugate
                HR = jnp.insert(jnp.conj(jnp.flipud(HR[1:N2])), 0, HR[0])
                HR = HR.at[N1].set(jnp.abs(HR[N1+1]))
                #HR[N1] = jnp.abs(HR[N1+1])
            
            z1 = (HR * Y.T).T # Temporal convolution
            z1 = jnp.fft.ifft(z1, axis=0)[:int(N+2*dN), :]
            #if (rdx+(sgn==1)*K1==0): debug = HR

            for sdx in range(K2): # Note zero indexing
                HS = gen_corf_j(sv[sdx], M1, SRF, [sdx+BP+1, K2+BP*2]) 
                z1 = z1*HS # Frequency convolution
                
                R1v = jnp.fft.ifft(z1, M2) # Second inverse FFT
                if dM == 0:
                    #cr[sdx, rdx+(sgn==1)*K1, :, :] = R1v[:, dM:dM+M]
                    cr = cr.at[sdx, rdx+(sgn==1)*K1, :, :].set(R1v[:, dM:dM+M])
                else: 
                    raise NotImplementedError
    return cr

@partial(jit, static_argnums=1)
def gen_cort_j(fc, L, STF, PASS):
    '''
    The primary purpose is to generate 2, 4,
    8, 16, 32 Hz bandpass filter at sample rate ranges from, roughly
    speaking, 65 -- 1000 Hz. Note the filter is complex and non-causal.

    Input: 
    fc: characteristic frequency
    L: length of the filter; power of 2 is preferable. Not differentiable.
    STF: sample rate
    PASS: [idx, K]; if idx=1, lowpass; if 1<idx<k, bandpass; if idx=K, highpass
    
    Output: (bandpass) cortical temporal filter for various length and sampling rate.
    '''

    # Tonotopic axis
    t = jnp.linspace(0, L-1, L) / STF * fc
    h = jnp.sin(2*np.pi*t) * t**2 * jnp.exp(-3.5*t) * fc
    h = h - jnp.mean(h)
    
    H0 = jnp.fft.fft(h, 2*L) # n-point fft, N=2L
    A, H = jnp.angle(H0[:L]), jnp.abs(H0[:L])
    #maxi = jnp.argmax(H)
    H /= jnp.max(H)
        
    #H = H * jnp.exp(1j*A)
    H *= jnp.sin(A) + 1j*jnp.cos(A)
    return H

from functools import partial

@partial(jit, static_argnums=1)
def gen_corf_j(fc, L, SRF, KIND):
    # if KIND == None:
    #     KIND = 2
    # if len(KIND) == 1:
    #     PASS = [2, 3]
    # else:
    #     PASS = KIND
    #     KIND = 2

    # tonotopic axis
    R1 = jnp.array([i for i in range(L)]) / L * SRF / 2 / jnp.abs(fc)
    if KIND == 1:
        C1 = 1/2/0.3/0.3
        H = jnp.exp(-C1*(R1-1)**2) + jnp.exp(-C1*(R1+1)**2)
    else:
        R1 = R1**2
        H = R1 * jnp.exp(1-R1)

    # Bandpass filtering
    #maxi = jnp.argmax(H)
    sumH = jnp.sum(H)
    # if PASS[0] == 1: #BPF
    #     H = H.at[:maxi].set(1)
    # elif PASS[0] == PASS[1]: # HPF
    #     H = H.at[maxi:L].set(1)
    H = H / jnp.sum(H) * sumH
    return H

@jit
def strf(y, sr):
  '''
  Python version of aud2cor() in the NSL toolbox. From auditory spectrogram to cortical STRF.

  Input:
  y: auditory spectrogram, duration x channels
  paras: parameters that generated the auditory spectrogram, e.g. [5, 8, -2, 0]
  rv: rate vector
  sv: scale vector
  out_filename: write the output in this file
  disp: normalization during display
  '''
  paras = [5, 8, -2, 0]
  STF, SRF = 1000/paras[0], 24
  
  N, M = y.shape
  N1, M1 = 2**ceil(np.log2(N)), 2**ceil(np.log2(M))
  N2, M2 = N1*2, M1*2
  
  Y = jnp.fft.fft(y, M2, axis=1)[:,:M1] # Fourier transform (frequency)
  Y = jnp.fft.fft(Y[:N,:], N2, axis=0) # Fourier transform (temporal)

  cr = []
  for i in range(len(sr)): # rate filtering
    (s,r) = sr[i,:]
    
    HR = gen_cort_strf(r, N1, STF) 
    z1 = (HR * Y.T).T # Temporal convolution
    z1 = jnp.fft.ifft(z1, axis=0)[:N, :]

    HS = gen_corf_strf(s, M1, SRF) 
    z1 = z1*HS # Frequency convolution
    R1v = jnp.fft.ifft(z1, M2) # Second inverse FFT
    cr.append(jnp.expand_dims(R1v[:, :M], axis=0))
  cr = jnp.vstack(cr)
  return cr

@partial(jit, static_argnums=1)
def gen_cort_strf(r, L, STF):
    '''
    Generate rate filter for rate r. Note the filter is complex and non-causal.
    The filter is padded such that half of it is zero. 

    Input: 
    r: characteristic frequency (of rate). Originally named fc.
    L: length of the filter; power of 2 is preferable. Not differentiable.
    STF: sample rate of the auditory spectrogram. If frmlen=5, then 200 Hz
    
    Output: (bandpass) cortical temporal filter for various length and sampling rate.
    '''
    #eps = 1e-15
    # Tonotopic axis
    t = jnp.linspace(0, L-1, L) / STF * jnp.abs(r) # Generating an array of frequencies
    t = jnp.sin(2*np.pi*t) * t**2 * jnp.exp(-3.5*t) * jnp.abs(r)
    
    t = jnp.fft.fft(t-jnp.mean(t), 2*L)
    #signs = jnp.concatenate([jnp.ones(1), jnp.ones(L-1) * jnp.sign(r), -jnp.ones(1), jnp.ones(L-1) * (-jnp.sign(r))])
    signs = jnp.concatenate([jnp.ones(L) * jnp.sign(r), jnp.ones(L) * (-jnp.sign(r))])
    signs = (signs+1)/2
    t = t.at[:].multiply(signs)
    #t = jnp.where(jnp.abs(t)<eps, eps, t)
    H = t / jnp.max(jnp.abs(t))
    return H

@partial(jit, static_argnums=1)
def gen_corf_strf(fc, L, SRF):

    # tonotopic axis
    R1 = jnp.array([i for i in range(L)]) / L * SRF / 2 / jnp.abs(fc)
    R1 = R1**2
    H = R1 * jnp.exp(1-R1)
    return H

###############################################################
# Some utility functions for training NNs with STRF as frontend

def initialize_sr(n_strfs, seed, method='random',
                  scale_cap=9, rate_cap=9):
  '''
  Initialize the new version of STRFs, where temporal modulation can be negative. 
  '''
  if method == 'random':
    np.random.seed(seed)
    s = np.random.rand(n_strfs)*scale_cap
    v = (np.random.rand(n_strfs)-0.5)*rate_cap*2
    sr = np.stack([s,v]).T
  elif method == 'log':
    if n_strfs == 40: 
      rv = 2**np.linspace(1, 5, 4)
      sv = 2**np.linspace(-2, 3, 5)
      sr = [[s, r] for s in sv for r in rv] + [[s, -r] for s in sv for r in rv]
      sr = np.stack(sr)
    else: 
      raise NotImplementedError
  else: raise NotImplementedError

  return sr


def initialize_compression_params(n_channels=129, val=1.0):
  '''
  Initialize compression parameters, one per channel.s 
  '''
  return jnp.ones(n_channels)*val