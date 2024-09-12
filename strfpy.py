from math import ceil, floor
from scipy.signal import convolve, lfilter

import numpy as np

def read_cochba():
    with open('/fs/clip-realspeech/projects/diffaud/cochba.txt') as f:
        cochba = f.readlines()
    out = []
    for line in cochba:
        line = line.strip("\n").split("\t")
        if line == "": continue
        out.append([])
        for num in line:
            num = num.replace("i","j")
            num = num.replace(" ","")
            out[-1].append(complex(num))
    return np.array(out)

def sigmoid(y, fac):
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
        y = np.exp(-y/fac)
        y = 1/(1+y)
    elif fac == 0:
        y = (y > 0) # Do I need to turn this into integer?
    elif fac == -3:
        raise NotImplementedError # halfregu()
    return y


def wav2aud(x, frmlen, time_constant, fac, octave_shift, filt_type='p', use_scipy=True, return_stage=5):
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
    global v5
    
    if filt_type != 'p': raise NotImplimentedError

    cf = np.linspace(-31, 97, 31+97+1)/24
    cf = [440*2**f for f in cf]
    cf = [round(f/10)*10 for f in cf]
    
    cochba = read_cochba()
    L, M = cochba.shape
    L_x = len(x)
    L_frm = round(frmlen * 2**(4+octave_shift)) # frame length (points)
    N = ceil(L_x/L_frm) # number of frames
    x = np.pad(x, (0,N * L_frm-len(x)), mode='constant') # zero-padding
    
    if time_constant != 0 : ### Still not sure if this is right? 
        alpha = np.exp( -1/ (time_constant * 2**(4+octave_shift)) )
    else:
        alpha = 0
    
    haircell_tc = 0.5 # hair cell time constant
    beta = np.exp(-1/ (haircell_tc * 2**(4+octave_shift)))

    # Bandpass filter and nonlinearities
    v5 = []
    ps = [p.real for p in list(cochba[0, :])] # beware of 0-indexing!
    for ch, p in enumerate(ps):
        temp = cochba[1:int(p)+2,ch]
        B, A = [f.real for f in temp], [f.imag for f in temp]
        if use_scipy:
            y = lfilter(B, A, x) # -> y1
        else:
            y = recursive_iir(B, A, x) # -> y1
        y = sigmoid(y, fac) # -> y2
        v5.append(y)
    v5 = np.vstack(v5)
    if return_stage==2: return v5
    
    # Apply a first difference filter -> y3
    v5 = v5[:-1, :] - v5[1:, :]
    if return_stage==3: return v5
        
    # Half wave rectifier -> y4
    v5 = np.maximum(v5, 0)
    if return_stage==4: return v5
    
    if alpha: # leaky integration ,-> y5
        if use_scipy:
            v5 = lfilter([1], np.array([1, -alpha]), v5) 
        else:
            out = []
            for ch in range(len(v5)):
                out.append(recursive_iir([1], np.array([1, -alpha]), v5[ch,:]))
            v5 = np.vstack(out)
        inds = np.arange(1, N+1)*L_frm-1
        #print(v5.shape, inds[-1])
        v5 = v5[:,inds]
    elif L_frm == 1: 
        pass
    else:
        raise NotImplementedError
    if return_stage==5: return v5
    #return [v5, cf]

def recursive_iir(b, a, x):
    '''
    Transfunction: b[0] + b[1]z^{-1} + ... / (a[0] + a[1]z^{-1} + ...)
    In time domain: a[0] y[n] = b[0]x[n] + b[1]x[n-1] + ... + b[m]x[n-m] - a[1]y[n-1] - ... - a[o]y[n-o]

    Assume that the initial condition is rest.
    x: length * dimensions. If the second dimension exists, the same filter would be applied along each dimension.
    '''
    
    assert a[0] == 1
    a1 = a[1:]
    ar = convolve(x, b) # AR part
    #print(ar.shape, x.shape)
    y = np.zeros(len(x)+len(a1))
    #return ar
    for i in range(len(x)):
        if len(a) > 1:
            ma = np.sum(y[i:i+len(a1)]* a1[::-1])
        else:
            ma = 0
        y[i+len(a1)] = ar[i] - ma
    return y[len(a1):]

def aud2cor(y, paras, rv, sv):
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

    dM = floor(M/2*FULLX) # Frequency index, =0
    dN = floor(N/2*FULLT) # Temporal index, =0
    if dM != 0 or dN != 0: 
        raise NotImplementedError
    N1, M1 = 2**ceil(np.log2(N)), 2**ceil(np.log2(M))
    N2, M2 = N1*2, M1*2
    
    Y = np.fft.fft(y, M2, axis=1)[:,:M1] # Fourier transform (frequency)
    Y = np.fft.fft(Y[:N,:], N2, axis=0) # Fourier transform (temporal)

    cr = np.zeros([K2, K1*2, N+2*dN, M+2*dM],dtype=complex)
    for rdx in range(K1): # rate filtering
        HR = gen_cort(rv[rdx], N1, STF, [rdx+1+BP, K1+BP*2])

        for sgn in [1, -1]:
            if sgn == 1: 
                HR = np.concatenate([HR, np.zeros(N1)])
            else: # conjugate
                HR = np.insert(np.conj(np.flipud(HR[1:N2])), 0, HR[0])
                HR[N1] = np.abs(HR[N1+1])
            
            z1 = (HR * Y.T).T # Temporal convolution
            z1 = np.fft.ifft(z1, axis=0)[:N,:]
            if (rdx+(sgn==1)*K1==0): debug = HR

            for sdx in range(K2):
                HS = gen_corf(sv[sdx], M1, SRF, [sdx+BP+1, K2+BP*2]) 
                cr[sdx, rdx+(sgn==1)*K1, :, :] = np.fft.ifft(z1*HS, M2)[:,:M] # Second inverse FFT
    return cr

def gen_cort(fc, L, STF, PASS):
    '''
    The primary purpose is to generate 2, 4,
    8, 16, 32 Hz bandpass filter at sample rate ranges from, roughly
    speaking, 65 -- 1000 Hz. Note the filter is complex and non-causal.

    Input: 
    fc: characteristic frequency
    L: length of the filter; power of 2 is preferable
    STF: sample rate
    PASS: [idx, K]; if idx=1, lowpass; if 1<idx<k, bandpass; if idx=K, highpass
    
    Output: (bandpass) cortical temporal filter for various length and sampling rate.
    '''

    if PASS == None: PASS = [2,3]

    # Tonotopic axis
    t = np.array([i for i in range(L)]) / STF * fc
    h = np.sin(2*np.pi*t) * t**2 * np.exp(-3.5*t) * fc
    h = h - np.mean(h)
    H0 = np.fft.fft(h, 2*L) # n-point fft, N=2L
    A, H = np.angle(H0[:L]), np.abs(H0[:L])
    maxi = np.argmax(H)
    maxH = H[maxi]
    H /= maxH

    # Passband
    # Might need to generally debug the indexing problems here
    if PASS[0] == 1: # LPF
        H[:maxi] = 1 #np.ones(maxi)
    elif PASS[0] == PASS[1]: 
        H[maxi:L] = 1 # np.ones(L-maxi, 1)
    H = H * np.exp(1j*A)
    return H


def gen_corf(fc, L, SRF, KIND):
    if KIND == None:
        KIND = 2
    if len(KIND) == 1:
        PASS = [2, 3]
    else:
        PASS = KIND
        KIND = 2

    # tonotopic axis
    R1 = np.array([i for i in range(L)]) / L * SRF / 2 / np.abs(fc)
    if KIND == 1:
        C1 = 1/2/0.3/0.3
        H = np.exp(-C1*(R1-1)**2) + np.exp(-C1*(R1+1)**2)
    else:
        R1 = R1**2
        H = R1 * np.exp(1-R1)

    # Bandpass filtering
    maxi = np.argmax(H)
    maxH = H[maxi]
    if PASS[0] == 1: #BPF
        sumH = np.sum(H)
        H[:maxi] = 1
        H = H / np.sum(H) * sumH
    elif PASS[0] == PASS[1]: # BPF
        sumH = np.sum(H)
        H[maxi:L] = 1
        H = H / np.sum(H) * sumH;
    return H

def corfftc(z, Z_cum, N, M, N1, M1, N2, M2, dN, dM, HR, HS, HH):
    if dN != 0:
        raise NotImplementedError
    z = np.pad(z, [(0, 0), (0,M2-z.shape[1])], mode='constant')

    Z = np.fft.fft(z, axis=1)[:,:M1]
    Z = np.pad(Z, [(0,N2-Z.shape[0]), (0,0)])
    Z = np.fft.fft(Z, axis=0)

    R1 = np.matmul(np.expand_dims(HR, axis=1), np.expand_dims(HS, axis=0))
    HH = HH + R1 * np.conj(R1)
    Z_cum += R1 * Z
        
    return Z_cum, HH
    
def cor2aud(cr, paras, rv, sv, N, M):
    BP = 0
    K1, K2 	= len(rv), len(sv)
    N1, M1 = 2**ceil(np.log2(N)), 2**ceil(np.log2(M))
    N2, M2 = N1*2, M1*2
    FULLT, FULLX = 0.0, 0.0
    para1 = np.concatenate([paras, np.array([FULLT, FULLX])])
    STF = 1000/para1[0]
    SRF = 24
    dM, dN = 0, 0 
    HH = 0
    Z_cum = 0
    for rdx in range(K1):
        HR = gen_cort(rv[rdx], N1, STF, [rdx+1+BP, K1+BP*2])

        for sgn in [1, -1]:
            if sgn == 1: 
                HR = np.conj(np.concatenate([HR, np.zeros(N1)]))
            else: # conjugate
                HR = np.insert(np.conj(np.flipud(HR[1:N2])), 0, HR[0])
                HR[N1] = np.abs(HR[N1+1])

            for sdx in range(K2): # Note zero indexing
                HS = gen_corf(sv[sdx], M1, SRF, [sdx+BP+1, K2+BP*2]) 
                z = np.squeeze(cr[sdx, rdx+(sgn==1)*K1, :, :])
                Z_cum, HH = corfftc(z, Z_cum, N, M, N1, M1, N2, M2, dN, dM, HR, HS, HH)

    # Normalize for DC
    HH[:, 0] = HH[:,0]*2
    #return Z_cum
    yh = cornorm(Z_cum, HH, N, M, N1, M1, N2, M2, dN, dM, NORM=0.9)
    return yh, HH

def cornorm(Z_cum, HH, N, M, N1, M1, N2, M2, dN, dM, NORM):
    #FOUTT, FOUTX = 0, 0

    # Modify overall transfer function
    sumH = np.sum(HH)
    HH = NORM * HH + (1-NORM) * np.max(HH)
    HH = HH / np.sum(HH) * sumH
    Z_cum = Z_cum / HH # Normalization

    y = np.fft.ifft(Z_cum, axis=0)[:N, :] # First ifft
    yh = np.fft.ifft(y, M2, axis=1)[:, :M] # Second ifft
    return yh*2