# Resilient Deep Distributed Neural Networks (RDDNNs)
We present multiple methods to retain accuracy in the case of single or multiple layer failure during inference in distributed neural networks.

## Methods
deepFogGuard -- Residual connections using our heuristic  
deepFogGuard+ -- Networks of stochastic depth, with failure of layers probability reflecting failure during inference.  

## Multiview Camera Object Classification Experiment
1. Obtain the already preprocessed [train](https://anonymousfiles.io/3GVNxBrV/), [test](https://anonymousfiles.io/PF7jVmsN/), and [holdout](https://anonymousfiles.io/wPsdRDxB/) datasets. Move the datasets to a single location and unzip. There should be three directories, one for each set. Modify directory paths in each method train file, near the top of file.
2. Install dependencies:
    * Tensorflow 1.8.0
    * Numpy >= 1.14.0
3. Train each method by running method\_train.py
4. To test our metric defined in the paper, weighted average accuracy, run failure\_iteration.py but change the first line of the file to restore the method you would like to test. For example, to test the baseline accuracy, the first line would instead read `from restore_baseline import test`.
5. Utilize batch run to compute performance over multiple iterations.

## Keras Single-lane Experiment (Health Activity and Cifar-10)
Dependencies:
   * Tensorflow 1.12.0
   * Keras 2.24
   * Keras-Applications 1.0.6
   * Keras-Preprocessing 1.0.5
   * Numpy 1.15.4
   * scikit-learn 0.20.1
