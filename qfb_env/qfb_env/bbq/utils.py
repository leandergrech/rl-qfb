import os
import numpy as np
import scipy as sc
import h5py as h
import pickle as pkl
from . import settings
from collections import defaultdict
from scipy.signal import filtfilt
from sklearn.gaussian_process.kernels import RBF

'''
Spectrum Simulation methods
'''
def magNormResponse(omegas, res, zeta):
    # Find corrected resonant frequency
    fixed_res = res / np.sqrt(1 - 2 * (zeta**2))
    
    # Create magnitude respose
    mag = (fixed_res**2.0) / np.sqrt(np.power(2.0 * omegas * fixed_res * zeta , 2.0) + np.power(fixed_res**2.0 - np.power(omegas, 2.0), 2.0))
    
    # Normalise
    mag = normalize(mag)
    
    return mag
    
def noiseHarmonics(freqs, freqs_harmonics, min_mag, max_mag):
    min_freq = freqs[0]
    max_freq = freqs[-1]
    
    n_f = len(freqs)
    noise = np.zeros(n_f)
    
    # Assign noise with random magnitude: mag ~ U(min_mag, max_mag) at each harmonic
    for harmonic in freqs_harmonics:
        if min_freq <= harmonic <= max_freq:
            idx = np.searchsorted(freqs, harmonic)
            noise[idx] = np.random.uniform(min_mag, max_mag, 1)
            
    # `Smear` the harmonics across adjacent freq bins
#     for i in np.arange(n_f)[1:-1]:
#          noise[i] = np.mean(noise[[i - 1, i, i + 1]])
    noise = filtfilt([1/3, 1/3, 1/3], 1, noise)
#     noise = filtfilt([1/4, 1/4, 1/4, 1/4], 1, noise)
    
    return noise

def getAxes(return_freqs=True, return_omegas=True):
    freq_res_mean = settings.freq_res_mean
    input_window_length = settings.input_window_length
    f_axis = settings.f_axis
    
    f_center_idx = np.searchsorted(settings.f_axis, freq_res_mean)
    f_min_idx = f_center_idx - int(input_window_length / 2)
    f_max_idx = f_center_idx + int(input_window_length / 2)
    
    f_axis_trunc = f_axis[f_min_idx: f_max_idx]
    o_axis = f_axis_trunc * 2 * np.pi
    
    if return_freqs and return_omegas:
        return f_axis_trunc, o_axis
    elif return_freqs:
        return f_axis_trunc
    elif return_omegas:
        return o_axis
    
'''
Fill data methods
'''
def getKeysFromBBQFiles():
    fillNb = 6890
    beamMode = 'FLATTOP'
    
    b1File = f"../data/Fill{fillNb}_B1_{beamMode}.h5"
    b2File = f"../data/Fill{fillNb}_B2_{beamMode}.h5"
    
    keys = defaultdict(dict)
    for filename in (b1File, b2File):
        with h.File(filename, 'r') as f:
            beamKey = getBeamKey(filename)
            for var in f.keys():
                varPattern = var.split(':')[-1].lower()
                keys[beamKey][varPattern] = var 
    return keys

def getACQ(fillNb=settings.fillNb, beamNb = settings.beamNb,
           plane = settings.plane, beamMode = settings.beamMode, basepath=os.path.join('..', 'data')):   
    
    keys = getKeysFromBBQFiles()
    filePattern = f'Fill{fillNb}_B{beamNb}_{beamMode}'
    for filename in os.listdir(basepath):
        if filePattern in filename:
            ext = os.path.splitext(filename)[-1]
            if 'pkl' in ext:
                with open(os.path.join(basepath, filename), 'rb') as f:
                    temp = pkl.load(f)
                    acq_data = np.array(temp[f'b{beamNb}'][f'acq_data_{plane}'])
                    acq_time = np.array(temp[f'b{beamNb}'][f'acq_data_{plane}_t'])
                    
                    return acq_data, acq_time
            elif 'h5' in ext:
                with h.File(os.path.join(basepath, f'Fill{fillNb}_B{beamNb}_{beamMode}.h5'), 'r') as f:
                    acq_data = np.array(f[keys[f'b{beamNb}'][f'acq_data_{plane}']])
                    acq_time = np.array(f[keys[f'b{beamNb}'][f'acq_data_{plane}_t']])
                    
                    return acq_data, acq_time
            else:
                assert False, f'File extention {ext} not recognised'

def getFFT(fillNb=settings.fillNb, beamNb = settings.beamNb,
           plane = settings.plane, beamMode = settings.beamMode, basepath=os.path.join('..', 'data')):
    
    acq_data, acq_time = getACQ(fillNb, beamNb, plane, beamMode, basepath)
    print(acq_data.shape)
    fft_data = mag(np.fft.rfft(acq_data * np.hanning(acq_data.shape[-1])))
    return fft_data, acq_time

def getIIR(fillNb=settings.fillNb, beamNb = settings.beamNb,
           plane = settings.plane, beamMode = settings.beamMode, basepath=os.path.join('..', 'data')):
    alpha = 0.5
    fft_data, fft_time = getFFT(fillNb, beamNb, plane, beamMode, basepath)
    iir_data = np.empty(fft_data.shape, np.float32)
    iir_data[0] = prev = fft_data[0]
    for i, fft in enumerate(fft_data[1:]):
        iir = alpha * fft + (1 - alpha)*prev
        iir_data[i + 1] = prev = iir
    return iir_data, fft_time

def getFitter(fillNb=settings.fillNb, beamNb = settings.beamNb,
              plane = settings.plane, beamMode = settings.beamMode, basepath=os.path.join('..', 'data')):
    
    fitterData = {}
    fitterTime = {}
    with open(os.path.join(f'{basepath}', f'Fill{fillNb}_B{beamNb}_{plane.upper()}_{beamMode}_fitter_freqs.pkl'), 'rb') as f:
        data = pkl.load(f)
        
        fitterData['min'] = np.asarray(data['min'])
        fitterData['max'] = np.asarray(data['max'])
        
        fitterTime['min'] = np.asarray(data['min_t'])
        fitterTime['max'] = np.asarray(data['max_t'])
        
    return fitterData, fitterTime

'''
Data modifier methods
'''
def clipFaxisToFitter(t, window_half_length, fitter_data, fitter_time, t_is_time=True):
    f_axis = settings.f_axis
    
    # Get index to retrieve fitter data
    if t_is_time:
        idx = np.searchsorted(fitter_time['min'], t)

        assert idx == np.searchsorted(fitter_time['max'], t), 'Fitter min and max are not synchronised'
    else:
        idx = int(t)
        
    if idx >= len(fitter_data['min']):
        idx = -1
    
    # Find window 
    min_freq = fitter_data['min'][idx]
    max_freq = fitter_data['max'][idx]
        
    center_idx = np.argmin(np.abs(f_axis - np.mean((min_freq, max_freq))))
    min_idx = center_idx - window_half_length
    max_idx = center_idx + window_half_length
    
    trunc_f_axis = f_axis[min_idx:max_idx].copy()
    
    return trunc_f_axis, min_idx, max_idx
    
def clipRealSpectrumToFitter(spectrum, t, window_half_length, t_is_time=True, basepath=os.path.join('..','data'), fillInfo=None):
    fitter_data, fitter_time = getFitter(**fillInfo, basepath=basepath)
    
    trunc_f_axis, min_idx, max_idx = clipFaxisToFitter(t, int(window_half_length), fitter_data, fitter_time, t_is_time)
    
    trunc_spectrum = spectrum[min_idx:max_idx].copy()
    
    return trunc_spectrum, trunc_f_axis

'''
Helper methods
'''
mag = lambda cplx: 20.0 * np.log10(np.abs(cplx))

normalize = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

standardize = lambda arr: (arr - np.mean(arr)) / np.std(arr)

getBeamKey = lambda filename: os.path.basename(filename).split('_')[1].lower()

is_cnn = lambda x: sum([1 if 'conv' in l.name else 0 for l in x._mylayers]) > 0

def getNbPointsInFFT(N):
    if N % 2 == 0:
        return int((N / 2) + 1)
    else:
        return int((N + 1) / 2)
    
'''
New Algorithms (IBIC)
'''
def tuneFinderPoly(freq, frame, deg):
    usable_indices = np.isfinite(frame)

    temp_freq = freq[usable_indices]
    temp_frame = frame[usable_indices]

    poly = np.poly1d(np.polyfit(temp_freq, temp_frame, deg=deg))
    fit = np.asarray([poly(f) for f in freq])

    tune = freq[np.argmax(fit)]

    return fit, tune

def tuneFinderFilter(sig, window_half_length):
    N = len(sig)
    sig_filt = np.zeros_like(sig)
    for i in range(N):
        cum_sum = 0
        ctr = 0
        for j in range(-window_half_length, window_half_length + 1):
            if ((i+j >= 0) and (i+j < N)):
                val = sig[i+j]
                if np.isfinite(val):
                    dist = window_half_length - np.abs(j)
#                     dist = alf ** np.abs(j)
                    cum_sum += val * dist
                    ctr += dist
        if ctr == 0:
            sig_filt[i] = sig[i]
        else:
            sig_filt[i] = cum_sum / ctr
        
    return sig_filt

def getIntersectionWindowSize(x1, x2, N=None, minInt=None, maxInt=None, tol=1e-10):
    assert isinstance(x1, list) or isinstance(x1, np.ndarray), "x1 must be of list type"
    assert isinstance(x2, list) or isinstance(x2, np.ndarray), "x2 must be of list type"
    assert len(x1) > 0, "Length of x1 is 0"
    assert len(x2) > 0, "Length of x2 is 0"

    if minInt is None:
        minInt = 1
    if maxInt is None:
        maxInt = min(N, len(x1))
    assert minInt < maxInt, "minimum intersection size must be smaller than maximum intersection size"

    intersectionWindowSize = 0
    for win_sz in np.arange(minInt, maxInt):
        if np.sum(np.abs(np.subtract(x1[-win_sz:], x2[:win_sz]))) <= tol:
            return win_sz
    raise Exception("No intersection found")

def GP(X1, y1, X2, kernel):
    '''
    Calculate posterior mean and covariance matrix for y2 based on the corresponding 
    input X2, the observations (y1, X1), and the prior kernel function.
    '''
    Sigma11 = kernel(X1, X1)
    Sigma12 = kernel(X1, X2)
    
    solved = np.linalg.pinv(Sigma11).dot(Sigma12).T
#     solved = sc.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
    mu2 = solved @ y1   # Posterior mean
    Sigma22 = kernel(X2, X2) # Posterior covariance
    Sigma2 = Sigma22 - (solved @ Sigma12)
    
    return mu2, Sigma2

def GP_noise(X1, y1, X2, kernel, sigma_noise):
    '''
    Calculate the posterior mean and covariance matrix for y2 based on the corresponding
    input X2, the noisy observations (y1, X1), and the prior kernel'''
    # Kernel of noisy observations
    Sigma11 = kernel(X1.reshape(-1,1), X1.reshape(-1,1)) + sigma_noise * np.eye(X1.size)
    # Kernel of observations vs to-predict
    Sigma12 = kernel(X1.reshape(-1,1), X2.reshape(-1,1))
    
    # Solve
    solved = np.linalg.pinv(Sigma11).dot(Sigma12).T
    #     solved = sc.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
    # Compute posterior mean
    mu2 = solved @ y1
    # Compute posterior covariance
    Sigma22 = kernel(X2.reshape(-1,1), X2.reshape(-1,1))
    Sigma2 = Sigma22 - (solved @ Sigma12)
    
    return mu2, Sigma2   

def tuneFinderGP(X1_raw, y1_raw, X2, kernel, noise_std=None, return_standardisation=False):
    if len(y1_raw.shape) < 2:
        y1_raw = y1_raw.reshape(1, -1)
        
    usable_indices = np.isfinite(y1_raw[0])  # Always the same columns are NaNs
    
    # Remove NaNs
    X1_finite = X1_raw[usable_indices]
    y1_finite = y1_raw[:, usable_indices]
    
    # Find std and mean of each frame and find the average std and average mean of each frame
    sigma_standardisation = np.mean(np.std(y1_finite, axis=1))
    mu_standardisation = np.mean(np.mean(y1_finite, axis=1))
    
#     y1_real = y1_real.flatten()
#     X1_real = np.repeat([X1_real], GP_nb_frames, axis=0).flatten()
    
    #***********************************************************************************
    # Standardise
    y1_standardised = (y1_finite - mu_standardisation) / sigma_standardisation
    
#     fig, (ax1, ax2) = plt.subplots(2)
#     for frame, frame_standardised in zip(y1_finite, y1_standardised):
#         ax1.plot(X1_finite, frame, marker='x', ls='None')
#         ax2.plot(X1_finite, frame_standardised, marker='x', ls='None')
#     ax1.set_title('Observed data')
#     ax2.set_title('Standardised data')
#     plt.show()
#     return
    
    # Get statistical info over multiple frames
    if noise_std is None:
        y1_noise = np.var(y1_standardised, axis=0)
    else:
        y1_noise = (noise_std/sigma_standardisation)**2
        
    y1_mean = np.mean(y1_standardised, axis=0)
#     print(y1_noise.shape)
#     print(y1_mean.shape)
    
#     fig, ax = plt.subplots()
#     ax.plot(np.repeat([X1_finite], GP_nb_frames, axis=0).flatten(), y1_standardised.flatten(), alpha=0.2, marker='x', c='k', ls='None')
#     ax.plot(X1_finite, y1_mean, c='tab:cyan', alpha=0.3)
#     ax.fill_between(X1_finite, y1_mean - y1_noise, y1_mean + y1_noise, color='tab:cyan', alpha=0.2)
    
    # Fit the GP
    gp_mu, gp_Cov = GP_noise(X1_finite, y1_mean, X2, kernel, y1_noise)
    gp_sigma = np.sqrt(np.diag(gp_Cov))

#     ax.plot(X2, gp_mu, 'tab:red')
#     ax.fill_between(X2, gp_mu - 1*gp_sigma, gp_mu + 1*gp_sigma, color='tab:red', alpha=0.3)
#     ax2 = ax.twinx()
#     ax2.plot(X2, gp_mu/gp_sigma, c='tab:green')
#     plt.plot()
#     return
    
    # Un-standardise    
    Mu = (gp_mu.flatten() * sigma_standardisation) + mu_standardisation
    Sigma = gp_sigma * sigma_standardisation
    #***********************************************************************************
    
    # Find tune wrt. mu and sigma
#     weighted_fit = np.divide(Mu, Sigma)
#     tune = X2[np.argmax(weighted_fit)]
    tune = X2[np.argmax(Mu)]
    
#     print(gp.kernel_)
    
    if return_standardisation:
        return Mu, Sigma, tune, mu_standardisation, sigma_standardisation
    else:
        return Mu, Sigma, tune
    
def tuneFinderGPLessParams(mag, freqs, len_scale, noise_std=None):
    k = RBF(len_scale)
    return tuneFinderGP(freqs, mag, freqs, k, noise_std)

def exponentiated_quadratic(xa, xb):
    sq_norm = -0.5 * sc.spatial.distance.cdist(xa, xb, 'sqeuclidean') * 1e-4
    return np.exp(sq_norm)

'''
Current algorithm Python implementation replicated from BQBBQLHC FESA class
'''
def medianFilter(input_array, nFFT, filter_core):
    filter_array = np.empty(2*filter_core, dtype=np.float)
    output_array = np.empty(nFFT, dtype=np.float)
    
    for i in np.arange(0, nFFT):
        count = 0
        for j in np.arange(0, 2*filter_core + 1):
            k = i - filter_core + j
            if k < 0:
                k = 0
            elif k >= nFFT:
                k = nFFT - 1
            
            if k != i:
                filter_array[count] = input_array[k]
                count += 1
            
        filter_array = np.sort(filter_array)
        output_array[i] = (filter_array[filter_core] + filter_array[filter_core - 1]) / 2
    
    return output_array

def avgFilter(input_array, nFFT, filter_core):
    output_array = np.empty(nFFT, dtype=np.float)
    
    filterLength = 2*filter_core + 1
    if filterLength <= 0:
        return input_array
    
    for i in np.arange(0, nFFT):
        avgValue = 0.0
        for j in np.arange(0, filterLength):
            k = i - filter_core + j
            if k < 0:
                k = 0
            elif k >= nFFT:
                k = nFFT - 1
            
            avgValue += input_array[k]
        
        avgValue /= filterLength
        output_array[i] = avgValue
    
    return output_array

def GaussFrequencyEstimate(data, dataLength, index):
    tresolution = 1. / (dataLength * 2)
    if index > 0 and index < dataLength - 1:
        left = data[index - 1]
        center = data[index]
        right = data[index + 1]
        
        val = index
        if left <= center and right <= center:
#             0.5 * std::log(right / left) / std::log(std::pow(center, 2) / (left * right))
            correctionTerm = 0.5 * np.log(right / left) / np.log(center * center / (left * right))
            if np.abs(correctionTerm) > 0.5:
                correctionTerm = 0.0
            val += correctionTerm
#             val *= tresolution
        return val
#     else:
#         val = index * tresolution
    raise Exception('Something is wrong with GaussFrequencyEstimate')
    
dBtoLin = lambda val: np.power(10, val / 20)

def fitTuneMedian(mag_reg_orig,
                  signal_raw,
                  nFFT,
                  fbinnedfrequency, 
                  freqMinLimit, 
                  freqMaxLimit, 
                  medianFilterLength, 
                  meanFilterLength,
                  refine,
                  finalEstimateIndexWidth, 
                  averageSpectra):
    
    signal_filtered = np.empty(nFFT, dtype=np.float)
    
    # Revolution frequency
    F_s = 11245.55
    
    medianFilterOutput = medianFilter(mag_reg_orig, nFFT, medianFilterLength)
    
    
    if averageSpectra:
        avgFilterOutput = avgFilter(medianFilterOutput, nFFT, meanFilterLength)
        signal_filtered = dBtoLin(avgFilterOutput)
    else:
        signal_filtered = dBtoLin(medianFilterOutput)
        
    # Find tune index -- first estimate
    maximum = tune_index = -1
    for i in np.arange(10, nFFT - 10):
        if fbinnedfrequency[i] >= freqMinLimit and fbinnedfrequency[i] <= freqMaxLimit:
            if signal_filtered[i] > maximum:
                maximum = signal_filtered[i]
                tune_index = i
    # tune_index holds the peak with the highest amplitude in the median (+avg) filtered spectra
    
    if refine:
        if tune_index != -1 and (tune_index - medianFilterLength) >= 0 and (tune_index + medianFilterLength) < nFFT:
            refinedTuneIndex = tune_index
            maximum = -1
            for i in np.arange(tune_index - finalEstimateIndexWidth, tune_index + finalEstimateIndexWidth + 1):
                if fbinnedfrequency[i] >= freqMinLimit and fbinnedfrequency[i] <= freqMaxLimit:
                    if signal_raw[i] > maximum:
                        maximum = signal_raw[i]
                        refinedTuneIndex = i
            
            fitted_index = GaussFrequencyEstimate(signal_raw, nFFT, refinedTuneIndex)
            fittedtune = (fitted_index / nFFT)*(freqMaxLimit - freqMinLimit) + freqMinLimit
#         else:
#             fittedtune = 0.0
#     else:
#         fittedtune = GaussFrequencyEstimate(signal_raw, nFFT, tune_index)
    else:
        raise Exception('bubu')
        
#     fig, ax = plt.subplots()
#     delt = f_axis[1] - f_axis[0]
#     ax.plot(f_axis, mag_reg_orig, marker='x', label='Spectrum')
#     ax.plot(f_axis, signal_raw, marker='x', label='raw')
# #     ax.plot(f_axis, medianFilterOutput, label='Median')
#     ax.plot(f_axis, avgFilterOutput, marker='x', label='Average')
#     ax.plot(f_axis, utils.dBtoLin(avgFilterOutput), marker='x', label='linear Average')
#     ax.axvline(f_axis[tune_index], color='k', label='tune_index')
#     ax.axvline(f_axis[refinedTuneIndex], color='b', ls='dashed', label='refined tune_index')
#     ax.axvline(fittedtune, color='g', label='fitted tune')
#     plt.show()
    
#     ax.legend()
    
#     exit(23)
        
    return fittedtune

def fitTuneMedianLessParams(mag, freqs):
    return fitTuneMedian(mag, dBtoLin(mag), len(freqs), freqs, freqs[0], freqs[-1], 5, 5, True, 1, True)
