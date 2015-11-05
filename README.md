# SCVB0-LDA is a multi-threading implementation of the SCVB0 algorithm in 
@inproceedings{Foulds:2013:SCV:2487575.2487697,
   author = {Foulds, James and Boyles, Levi and DuBois, 
   Christopher and Smyth, Padhraic and Welling, Max},
   title = {Stochastic Collapsed Variational Bayesian Inference 
            for Latent Dirichlet Allocation},
  }

The SCVB0 algorithm is implemeted in C++ and relies heavily on the EIGEN matrx library, 
which is included in the include/ folder, and on OpenMP for multi-thread programming.

Quick usage:
  1. make
  2. ./run.sh
  
This project is the mini-project required in CSCI686 Fall 2015 in the University of Southern California.
