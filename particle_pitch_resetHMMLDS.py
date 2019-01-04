import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import scipy.fftpack as fft

################################################################################
## Functions for use in main loop 
################################################################################
def make_template(f0, num_h, w_size, fs, alpha=0.5, sigma=0.03):
    '''
    Generates the values for a harmonic template of a specified pitch. Used as a
    compatibility function to determin emission probabilities
    '''
    df = fs/w_size
    freqs = np.arange(0, fs/2, df)
    assert len(freqs) == w_size/2
    template = np.zeros(int(w_size/2))
    for k, f in enumerate(freqs):
        for p in range(1, num_h+1):
            template[k] += ((1/p)**(alpha))*np.exp( -(f-p*f0)**2 / ( 2*(sigma*p*f0)**2 ))
    return template

def make_templates(f0_vals, num_h, w_size, fs, sigma=0.03):
    '''
    Generates a family of templates for the specified pitches f0_vals. See
    make_template for basic functionality.
    '''
    templates = np.zeros((len(f0_vals), int(w_size/2)))
    for k, f0 in enumerate(f0_vals):
        templates[k,:] = make_template(f0, num_h, w_size, fs, sigma)
    return templates

def discrete_sample(p_vals):
    '''
    takes an arbitrary discrete distribution and returns a randomly selected
    state.  works by dividing up the unit interval according to the given
    probabilities, sampling from a uniform distribution, and choosing the
    state based on into which bucket the uniform sample falls.
    '''
    cum = np.cumsum(p_vals)
    num_vals = len(cum)
    if np.abs(cum[-1]-1.) > 0.01:
        print(cum)
        print("Can't sample: not a valid distribution!")
        return 0
    old = 0
    u = np.random.uniform()
    for k in range(num_vals-1):
        if old <= u and u < cum[k]:
            return k
        old = cum[k]
    return num_vals-1 

def p_emission(spec, f0, num_h, w_size, fs, sigma=0.03):
    '''
    Returns the emission probability of a certain state (f0) given the
    observation (spec).  The "probability" is unnormalized!
    '''
    t = make_template(f0, num_h, w_size, fs, sigma)
    return np.dot(t, spec[:int(w_size/2)])

def particles_to_dist(particles, weights):
    '''
    Converts particles into a histrogram for resmapling
    '''
    assert len(particles) == len(weights)
    vals = np.unique(particles)
    probs = np.zeros_like(vals, dtype=float)
    for k, w in enumerate(weights):
        idx = np.where(vals==particles[k])
        probs[idx[0][0]] += w
    return vals, probs 

def particles_to_dist_bin(particles, weights):
    '''
    Converts particles and weights to an empirical distribution over the hidden
    states.  Here the hidden state is binary (both the note and the measure
    frequency).
    '''
    particles = [(particles[t,0],particles[t,1]) for t in range(len(particles))]
    vals = []
    probs = []
    for k, p in enumerate(particles):
        if p in vals:
            idx = vals.index(p)
            probs[idx] += weights[k]
        else:
            vals.append(p)
            probs.append(weights[k])
    return vals, probs

def particle_expectation(particles, weights):
    return np.dot(particles, weights)

def q_sample(particles, weights):
    '''
    Samples the current hidden state given the message in the form of particles and weights
    '''
    vals, probs = particles_to_dist(particles, weights)
    idx = discrete_sample(probs)
    return vals[idx]

def q_sample_bin(particles, weights):
    '''
    Samples the current hidden state given the message in the form of particles
    and weights. This version is for the two-component hidden state: note and
    estimated frequency.
    '''
    vals, probs = particles_to_dist_bin(particles, weights)
    idx = discrete_sample(probs)
    return vals[idx]

def s_sample(state_old):
    '''
    Sample from the note transition probability distribution given the previous note. 
    '''
    states =  np.array([state_old*((2**(1/12))**k) for k in range(-6,7)])
    probs = np.ones_like(states)
    mid_point = int(len(states)//2)
    probs[mid_point]*= 120
    probs[:mid_point] *= np.linspace(0.1,1,mid_point)*0.8
    probs[mid_point+1:] *= np.linspace(1,0.1,mid_point)*0.8
    probs /= np.sum(probs)
    idx = discrete_sample(probs)
    return states[idx]


################################################################################    
## set up basic audio parameters and test signal 
################################################################################    
fs = 44100
dt = 1./fs
w_size = 1024 
w_half = int(w_size/2)
w_t = w_size/fs
f_min = 1/w_t
df = fs/w_size
f_vals = np.arange(-fs/2, fs/2, df)
print(len(f_vals))
print(f_vals)
print("window size: {} samples \t {} seconds \t {} f low".format(w_size, w_t, f_min))

## test signal
dur = 0.2
num_h = 8
t = np.arange(0, dur, dt)
f0_offset = np.sin(2*np.pi*t*(1/(2*dur)))*10
f0 = np.ones_like(t)*220 + f0_offset*0
in_sig = np.zeros_like(t)
for p in range(1, num_h+1):
    in_sig += np.cos(2*np.pi*t*f0*p)

## Generate Signal: First notes of Bach's Die Kunst der Fuge with added noise
def gen_env(a, d, r, sus_val, dur_n, fs):
    t_a = np.arange(a*fs)/fs
    t_d = np.arange(d*fs)/fs
    t_r = np.arange(r*fs)/fs
    t_s_n = dur_n - len(t_a) - len(t_d) - len(t_r) 
    assert dur >= 0
    gamma = np.log(2)/a
    seg_a = np.exp(gamma * t_a) - 1
    gamma = -np.log(sus_val)/d
    seg_d = np.exp(-gamma*t_d)
    gamma = np.log(sus_val+1)/r
    seg_r = (sus_val+1) - np.exp(gamma*t_r)
    seg_s = np.ones(t_s_n)*sus_val
    return np.concatenate([seg_a, seg_d, seg_s, seg_r])
num_h = 10 
d = 293.6648; a = 440.0; f = 349.2282; cs = 277.1826; e = 329.6276;
dur = 0.200     # eighth note duration
f *= 1.01 # mistuning of the third -- make a little sharp
freqs = [d, a, f, d, cs, d, e, f]
durs = [4*dur, 4*dur, 4*dur, 4*dur, 4*dur, 2*dur, 2*dur, 5*dur]
notes = []
for k, freq in enumerate(freqs):
    num_cycles = int(durs[k]*freq)
    t = np.arange(0, num_cycles/freq, dt)
    note = np.zeros_like(t)
    for n in range(num_h):
        note += ((1/(n+1))**0.1)*np.cos(2*np.pi*freq*(n+1)*t)
    note = note * gen_env(0.02, durs[k] - 0.045, 0.02, 0.65, len(note), fs)
    notes.append(note)
in_sig = np.concatenate(notes)
in_sig /= np.max(in_sig)
in_sig += np.random.normal(scale=np.sqrt(0.0000001 * fs / 2), size=in_sig.shape)


################################################################################
## particle filtering
################################################################################
## initialize variables and first step
num_particles = 150  
step_size = int(w_size/2)
max_steps = int((len(in_sig)-w_size)//step_size)
# max_steps = 10
print("max steps:", max_steps)
particles = np.zeros((max_steps, num_particles, 2))
particles[0,:,0] = np.ones(num_particles)*d
particles[0,:,1] = np.ones(num_particles)*d
weights = np.zeros_like(particles[:,:,0])
weights[0,:] = np.ones(num_particles)*(1/num_particles)
sigma_factor = 0.009

## main filtering loop
for t in range(1, max_steps):
    print("step: ", t)
    spec = np.abs(fft.fft(
            np.hanning(w_size)*in_sig[(t-1)*step_size:(t-1)*step_size+w_size]))
    for k in range(num_particles):
        previous = q_sample_bin(particles[t-1,:,:], weights[t-1,:])
        particles[t,k,0] = s_sample(previous[0]) 
        if particles[t-1,k,0] == particles[t,k,0]:
            particles[t,k,1] = np.random.normal(previous[1],
                    previous[1]*sigma_factor)
        else:
            particles[t,k,1] = np.random.normal(particles[t,k,0],
                    particles[t,k,0]*sigma_factor)
        weights[t,k] = p_emission(spec, particles[t,k,1], 6, w_size, fs)
    weights[t,:] /= np.sum(weights[t,:])


################################################################################
## plot results
################################################################################
points = np.zeros((3, max_steps*num_particles))
map_points_s = np.zeros((max_steps))
map_points_f = np.zeros((max_steps))
map_points_f_expected = np.zeros((max_steps))
count = 0
for t in range(max_steps):
    #vals, probs = particles_to_dist_bin(particles[t,:,:], weights[t,:])
    vals, probs = particles_to_dist(particles[t,:,0], weights[t,:])
    idx = np.argmax(probs)
    state = vals[idx]
    map_points_s[t] = state 
    idcs = np.where(particles[t,:,0] == state)
    max_idx = np.argmax(weights[t,idcs]) 
    map_points_f[t] = particles[t,idcs,1][0][max_idx]
    map_points_f_expected[t] = np.dot(particles[t,idcs,1][0],
            weights[t,idcs][0])/np.sum(weights[t,idcs][0])
    # map_points_f[t] = vals[np.argmax(probs)][1]
    # map_points[t] = particles[t,np.argmax(weights[t,:]),1]
    for k in range(num_particles):
        points[0, count] = t
        points[1, count] = particles[t,k,1]
        points[2, count] = (25*weights[t,k])**2
        count += 1
plt.scatter(points[0,:], points[1,:], s=points[2,:])
plt.plot(map_points_s, color='k', linestyle='--')
plt.plot(map_points_f_expected, color='r', linestyle='-.')
plt.xlabel("Window Number (each window approximately 23 ms)")
plt.ylabel("Pitch Frequency (Hz)")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Block Number (approximately 12 ms each)", size=20)
plt.ylabel("Frequency (Hz)", size=20)
plt.ylim(100,800)
plt.show()
