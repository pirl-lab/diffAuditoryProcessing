from jax.scipy.signal import stft
from jax import numpy as jnp
from jax import jit, vmap

def spectrogram(x, window_length, sr=16000):
    _,_,y = stft(x, fs=sr, 
                 nperseg=window_length, 
                 noverlap=window_length//4)
    return y

def multiscale_spectrogram_loss(x, xh, window_lengths):
    loss = 0
    for window_length in window_lengths:
        s, sh = spectrogram(x, window_length), spectrogram(xh, window_length)
        loss += jnp.mean(jnp.abs(s.real-sh.real))
        loss += jnp.mean(jnp.abs(s.imag-sh.imag))
    return loss

def wavform_loss(x,xh):
    return jnp.mean(jnp.abs(x-xh))

@jit
def bsrnn_loss(x, xh, loss='L1'):
    if loss=='L1':
        spec_loss = multiscale_spectrogram_loss(x, xh, [256, 512, 1024])
        wav_loss = wavform_loss(x,xh)
    else: 
        raise NotImplementedError
    return spec_loss + wav_loss

vbsrnn_loss = vmap(bsrnn_loss)