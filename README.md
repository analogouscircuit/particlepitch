# Particle Pitch
Two scripts for showing how the addition of a simple note transition model to a 
[template-matching pitch model](https://github.com/analogouscircuit/pyscfb), by way of a particle filte,
can improve pitch
estimates.  The example used here is a synthesized rendition of the opening 
phrase of the first fugue from Bach's _Die Kunst der Fuge_.  Added noise
causes the simple pitch model to produce octave confusions.  The first model
uses a simple discrete state space Hidden Markov Model (HMM) with hand-coded
state (note) transition probabilities.  The second example, a switching Hidden
Markov Model + Linear Dynamic System model (in David Barber's terminology), allows
for a continuous range of possible pitch frequencies and uses a continous probability
distribution for note transitions.  Samples outputs shown below.

(1) Signal and "Ground Truth"
![Signal and GT](/images/signal_and_gt.png)


(2) Naive estimate (template matching only)
![naive](/images/naive.png)

(3) HMM estimate
![HMM](/images/HMM.png)

(4) Reset-HMM-LDS estimate
![Reset-HMM-LDS](/images/reset_HMM_LDS.png)

## References.

Barber, David (2012). _Bayesian Reasoning and Machine Learning_. (Cambridge University Press, Cambridge).
