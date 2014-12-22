AsyncFTRL
=========

A multithreaded implementation of FTRL-Proximal algorithm for Logistic Regression. This library is parallelized by an asynchronous procedure, followed the work of Downpour SGD by Jeffrey Dean etc.

Related Work:
 * FTRL: http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
 * Asynchronous SGD: http://research.google.com/archive/large_deep_networks_nips2012.html

## Features
 * LibSVM file format
 * Multithreaded accelerated

## Get Started
 * Single thread mode: ./ftrl_train -f input_file -m model_output [-t test_file]
 * Multithread mode: ./ftrl_train -f input_file -m model_output [-t test_file] --thread 0

## Play with Async FTRL
Most of the time async ftrl works pretty well and you don't need to touch async ftrl related parameters. But if dosen't work, you may try the following:
 * sync-step: number of push/fetch steps to sync up with global model, default is 3. you may try 2/1 if default param fails.
 * warmstarting: train a single model using a small fraction of the data before async ftrl start.
   - --burn-in fraction : set fraction of data used to train a single model before async ftrl start.
