# vrnn

### cut&paste:
- ssh -L 16006:127.0.0.1:6006 amlab
- scp rough_cut_200_pad_0_max_300_norm_xyonly.npy fharder@146.50.28.47:~/generative_audio_vrnn/vrnn_test/data/handwriting/
- scp fharder@146.50.28.47:~/generative_audio_vrnn/vrnn_test/data/logs/handwriting_/checkpoint .
- contrib training bucket by seq len
- tensorarray

###done:

- build VAE, run on mnist
- build VRNN, run on handwriting data

###in progress:
- try and get an actual mixture going
- test binary loss

###to do:
- extend to full length data
- consider correlated output dist
- report

###notes:
- reconsider gradient clipping and masking

###test log:
- test 08: +-1 clipping led to steady run over 5000 iterations... no idea why.
 spotted some discrepancies in the log p error, which should be investigated further.

- test 09 +- bound set to 100 at lr = 0.003 error at 1000 is 38523.2

- test 10 +- bound set to 1000 at lr = 0.0003 error at 600 is 41065.5 time: 2281.5763669

- test 11 no clipping lr = 0.003 error at 1000 is 38587.6 time: 3573.4575541

- test 12 no clipping lr = 0.01 error at 1000 is 38337.4 time: 3141.58930588 but convergence set in around 400-600 already

- test 13 no clipping added 200 hidden units to each net except lstm 

- test 14 found error normalizing data... redoing 13 .. and seemingly, there is no difference in training error??  -  generation (pix4-6) yields gibberish again, but look a bit more structured that first batch. and thankfully goes left to right, which is a start i guess (though that might be just the renorm)

- test 15 messed up masking.. needs redo

- test 16 still crap. slightly smaller error around 38000 though

- test 17 no masking test. lr = 0.003  - error went down to around 37100 - then got timed out. need to retrieve the 500 it save and check it.

- test 18 upped initial variance in sigmas - results in less movement in overall distribution, which remains vaguely gaussian and does not shift in mean.

- test 19 trying out scalar summaries for kldiv and log_p - lr=0.01 kldiv goes to 0 - log_p converges quickly to around - 38000

- test 20 only optimizing (-log_p) - still converges around 38000. simply does not learn identity mapping?

- test 21 set epsilon z noise to zero - same convergence behavior. fails to learn identity mapping. wtf is wrong here?

- test 22 increased variance and bias in output to normal variance weights. to no effect

- test 23 decreased lr to 0.0001 - convergence around 36700, so limited effect

- test 24 shrinking lstm size to 20 bc it should have no impact here - sanity check. it doesn't

- test 25 removing cov offset - checking if it still crashes after switching elus for relus way back. my guess is, it should, with cov going to 0. unless the elu gradient was actually the problem. did not blow up for some reason... but spikes in error could be explained by instability due to small variances. and having no latent noise does not restrict that variance, so it might still crash, once noise is reintroduced

- test 25, 26 used naive reconstruction error (x-y)^2 instead of log p loss. now if this doesn't go to 0, something must be wrong with the latent space i think. after 600 it. error is around 6/200 = 0.03 per timestep. seems fine. goes down to min of around 1.2/200 < 1%, but becomes highly unstable. smaller learning rate will likely fix this... anyway. it's clear enough, that this identity mapping is actually learned. 

- test 27 back to proper setup (loss, noise) and setting minimum variance to 1 - nothing much that is new. fairly nice convergence on the kldiv front. 

- test 28 fixed variance trial (to rule out exploding variance hypothesis) - curiously enough, log_p still converges at -37k. after 2k iterations, kldiv is at 6.5, which is promising. a longer run is needed to look at convergence to 0. (at 1k iterations, kldiv was around 950)

- test 29 fixed variance at 100 - led to worse log_p scores (37600), which suggests, that in test 28, the targets tend to lie relatively close to the estimate means and not in the log tail. well that, or it makes inference via latent variables too hard

- test 30 one more fixed variance test with 0.05 - given normalized data, maybe std = 1 was already too large. kldiv starts off in 10^5 but seems to minimize efficiently. log_p is worse and converges seemingly around 40k, though maybe not. really need to visualize actual values in the model.

- test 31 first run with gm - same scores initially. kldiv a lot higher but shrinking, log p at 36.7k 

- test 32 run plotting latent means and covariances. suggests that means go to 0.

- test 33 reintroduced elu's in all networks that generate dists. and the do make distributions, but thos seem to become stable after a while

- test 34 added output dist params to plots. spotted some odd behaviour with most means far away from 0. peak at -50, going as far as -350 (cutoff before norm was 300, no masking)

- test 35 went back to gaussian output. ouput dist certainly not as far from 0 but hard to make out

- test 36 bounded plots for debugs and gradients (latter needs more work) suggests that both latent and output distributions actually look kind of like what one would expect. cov_x does however grow larger than expected

- test 37 cov_x goes to around 7. 

- test 38 convergence over 5k iterations - kldiv fluctuates between 10 and 1000. lr adjustments might fix this. masking might also be at fault?

- test 39 redo of 38 without masking, gradient clipping and relus. kldiv converges to <0.02 in 3k iterations. log p unchanged

- test 40 passing parts of log_p for debug - passed wrong dimension (z=200 instead of x=2), creating huge offset to error function. no impact on gradients though

- test 41 first run with corrected log_p. high lr=0.003 for first results. after 300 iterations bound around 1k after 2.5k iterations bound around 600

- test 42 passing xdiff- does better than chance (which should be around 1) goes to around 130/200.

- test 43 optimizing logp - 2 short runs, then 2 long. 1) with noise convergence slows around 80-70/200, 2) without noise around 50, 3) with substantially smaller net, lr is upped a little again - does converge to something like 5/200= 0.025 average diff. around 6k steps. so yes to 0. 4) small net with noise: convergence behavior again rather nice, although slighty slower (duh) with around 12/200 at 6k steps. this verifies at least, that the identity mapping can indeed be learned. (restore 10 sanity)

- test 44 run original model with small weight setups. especially small latent dim. - aaaaaand IT'S ALIIIVE! first samples resembling handwriting! 10k iterations
assumed potential causes:
 + way lower learning rate (no overshoots)
 + small model (generally faster convergence and way more iterations possible)
 + reduced latent space (something something autoencoders? also reduced variance)
 + con: kl-div not converged (really slow towards the end)
 + con: limited variability due to small latent space

- test 45 next run with slightly larger architecture and higher lr - 

- test 46 loaded and continued with weights from 45. seems like it's already converged though

- test 47 setup as in 45 but with gm k=5

- test 48 setup zdim = 10

- test 49 binary gauss with zdim = 10, next run try zdim = 2
