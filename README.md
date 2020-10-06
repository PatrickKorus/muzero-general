# MuZero General Fork

This is the fork of the MuZero implementation at [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) which
closely follows the pseudo code attached to the [original paper](https://arxiv.org/abs/1911.08265v2). I have made some 
minor changes and made logging available in a `matplotlib` kind of fashion for our experiments described in  my thesis:

#### [An Evaluation of MCTS Methods for Continuous Control Tasks](https://github.com/PatrickKorus/muzero-general) 

The following docs are taken from [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) where
more documentation can be found. A detailed explanation of how MuZero works can also be found in my thesis.

## Code structure

![code structure](https://github.com/werner-duvaud/muzero-general/blob/master/docs/code-structure-werner-duvaud.png)

## Getting started
### Installation

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
cd muzero-general

pip install -r requirements.txt
```

### Run

```bash
python muzero.py
```
To visualize the training results, run in a new terminal:
```bash
tensorboard --logdir ./results
```

## Authors

* Werner Duvaud
* Aurèle Hainaut
* Paul Lenoir
* [Contributors](https://github.com/werner-duvaud/muzero-general/graphs/contributors)

* Minor changes for the thesis by me, Patrick Korus (patrick.korus@stud.tu-darmstadt.de)
