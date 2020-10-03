# RelaxedPredictiveCoding
Repo for experiments for the relaxed predictive coding paper

This paper shows that 3 constraints on the biological plausibility of predictive coding models (the weight transport problem, the backwards nonlinearities problem, and the 1-1 error connectivity, can be relaxed -- i.e. removed entirely or else ameliorated with additional parameter sets without unduly affecting performance on supervised learning tasks. We run experiments on MNIST and Fashion-MNIST with a 4-layer MLP model. 

## Installation and Usage
Simply `git clone` the repository to your home computer. The `main.py` file will run the main model. 

You should be able to replicate the experiments in the paper through the following commandline options `--use_backwards_weights True --update_backwards_weights True` will run the network with learnable backwards weights. To run with initially random and non-learned backwards weights (i.e. Feedback Alignment) use `--use_backwards_weights --updath e_backwards_weights False`. 

To run without backwards nonlinearities use the command line option `--use_backwards_nonlinearities False`. To run with full error connectivity, run `use_error_weights True`.

You can also specify the dataset ("mnist" or "fashion") and the activation function ("tanh","relu","sigmoid") via the commandline options `--dataset` and `--act_fn`.

## Requirements 

The code is written in [Python 3.x] and uses the following packages:
* [NumPY]
* [PyTorch] version 1.3.1
* [matplotlib] for plotting figures

If you find this code useful, please reference as:
```
@article{millidge2020relaxing,
  title={Relaxing the Constraints on Predictive Coding Models},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L and Seth, Anil},
  journal={arXiv preprint arXiv:2009.05359},
  year={2020}
}
```
