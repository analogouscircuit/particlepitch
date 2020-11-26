import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as dsp
import scipy.fftpack as fft

## functions
def make_template(f0, num_h, w_size, fs, sigma=0.03):
    df = fs/w_size
    freqs = np.arange(0, fs/2, df)
    assert len(freqs) == w_size/2
    template = np.zeros(int(w_size/2))
    for k, f in enumerate(freqs):
        for p in range(1, num_h+1):
            template[k] += ((1/p)**(0.1))*np.exp( -(f-p*f0)**2 / ( 2*(sigma*p*f0)**2 ))
    return template

def make_templates(f0_vals, num_h, w_size, fs, sigma=0.03):
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
    Converts particles into a histrogram/
    '''
    assert len(particles) == len(weights)
    vals = np.unique(particles)
    probs = np.zeros_like(vals, dtype=float)
    for k, w in enumerate(weights):
        idx = np.where(vals==particles[k])
        probs[idx[0][0]] += w
    idx = discrete_sample(probs)
    return vals, probs 

def particle_expectation(particles, weights):
    return np.dot(particles, weights)

def q_sample(particles, weights):
    assert len(particles) == len(weights)
    vals = np.unique(particles)
    probs = np.zeros_like(vals, dtype=float)
    for k, w in enumerate(weights):
        idx = np.where(vals==particles[k])
        probs[idx[0][0]] += w
    idx = discrete_sample(probs)
    return vals[idx]
    
def s_sample(state_old):
    '''
    Sample from the note transition probability distribution given the previous note. 
    '''
    states =  np.array([state_old*((2**(1/12))**k) for k in range(-7,8)])
    probs = np.ones_like(states)
    mid_point = int(len(states)//2)
    probs[mid_point]*= 160
    probs[:mid_point] *= np.linspace(0.1,1,mid_point)*0.8
    probs[mid_point+1:] *= np.linspace(1,0.1,mid_point)*0.8
    probs /= np.sum(probs)
    idx = discrete_sample(probs)
    return states[idx]


## basic params
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
## Die Kunst der Fuge, first notes
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

## particle filtering
num_particles = 100
step_size = int(w_size/2)
max_steps = int((len(in_sig)-w_size)//step_size)
particles = np.zeros((max_steps, num_particles))
particles[0,:] = np.logspace(np.log10(110), np.log10(440), num_particles)
particles[0,:] = [110*((2**(1/12))**(k%25)) for k in range(0, num_particles)]
weights = np.zeros_like(particles)
weights[0,:] = np.ones(num_particles)*(1/num_particles)

for t in range(1, max_steps):
    print("step: ", t)
    spec =np.abs(fft.fft(
            np.hanning(w_size)*in_sig[(t-1)*step_size:(t-1)*step_size+w_size]))
    for k in range(num_particles):
        previous_state = q_sample(particles[t-1,:], weights[t-1,:])
        # particles[t,k] = np.random.normal(previous_state, previous_state*.002)
        particles[t,k] = s_sample(previous_state)
        weights[t,k] = p_emission(spec, particles[t,k], 6, w_size, fs)
    weights[t,:] /= np.sum(weights[t,:])

## set up points and weights for scatter plot
points = np.zeros((3, max_steps*num_particles))
map_points = np.zeros((max_steps))
count = 0
for t in range(max_steps):
    vals, probs = particles_to_dist(particles[t,:], weights[t,:])
    map_points[t] = vals[np.argmax(probs)] 
    for k in range(num_particles):
        points[0, count] = t
        points[1, count] = particles[t,k]
        points[2, count] = (2*num_particles*weights[t,k])**2
        count += 1
times = np.arange(len(points[0,:]), dtype=float)
times *= step_size/fs
labels = ["{:.3f}".format(t) for t in times]
plt.scatter(points[0,:], points[1,:], s=points[2,:])#
plt.xlabel("Window Number (each window approximately 23 ms)")
plt.ylabel("Pitch Frequency (Hz)")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Block Number (approximately 12 ms each)", size=20)
plt.ylabel("Frequency (Hz)", size=20)

plt.plot(map_points, color='r')
plt.show()
