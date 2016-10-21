# vrnn
###done:

- build VAE, run on mnist
- build VRNN, test on artificial data-set

###in progress:

- set up read-then-generate model,
- find appropriate data-set,
- take look at data sets from rgp paper,

###to do:

- try and get RGP implementation running,
- think of decent model parameters, net sizes
- train for real
- run check
- parameter changes
- comparison
- report

###notes:
- s3 works with tanh and weighted softplus, indicating problems with relu
- s7 'works' with tanh. mean_x are at +-6, with cov_x at 9. no separation of random cases
- theory about relus: large x_diff pushes for increasing cov_x, 
which grows quickly to the point, where it's mostly flat in the range of x_target, 
thus decreasing gradients for x_mean, which never catches up. 
Clipping by global norm does its part. local norm should be better (needs investigation) 
