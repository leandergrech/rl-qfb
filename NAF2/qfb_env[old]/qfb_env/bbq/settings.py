import numpy as np

# Constants
F_s = 11245.55
N = 2048
Nf = 1025
f_axis = np.linspace(0, F_s/2, Nf)
delta_f = np.mean(np.diff(f_axis))


# Fill parameters
fillNb = 6890
beamNb = 1
plane = 'h'
beamMode = 'FLATTOP'

# Simulation parameters
input_window_length = 100
freq_res_mean = 3e3
zeta_uniform_low = -2
zeta_uniform_high = -1.5

# Harmonics parameters
Noise_harmonic = 50.0
Noise_half_drop = 5.5
Noise_std = 0.08