# ARIAC - Grand Challenge 1

Yann Claes<br/>Antoine Gratia<br/>Gaspard Lambrechts

## Hybridization of models and data towards an augmented engineering

This repository contains the codes for the different projects and use cases
achieved as part of the **Grand Challenge #1** of the **ARIAC project**.

## Milestone 1: Additive manufacturing

The first milestone concerned the usage of machine learning models in order
to predict the temperature in a metallic part during a simplified process of
additive manufacturing (two-dimensional problem, constant domain / no material
addition, simple laser one-dimensional displacement with only two parameters).
The goal is to avoid to rely on complex and costly simulations such as
finite-element methods.

The codes are available in the [milestone-1](./milestone-1) directory. Two main
models have been considered. First, a stationary model (MLP) can be trained
using `python3 mlp.py NAME SCENARIO` with `NAME` the name of the training
session and `SCENARIO` the data split scenario (see report). Second, a
recurrent model (RNN) can be trained using `python3 rnn.py NAME SCENARIO` with
the same parameters. A random forest can also be trained with `python3 rf.py`.

The trained models' prediction can be visualised using `python3 mlp.py NAME
SCENARIO` and `python3 rnn.py NAME SCENARIO` respectively.

This [short report](./milestone-1/short-report.pdf) formally describes the
problem and the methods, and displays the results of the different methods.
This [more extensive report](./milestone-1/long-report.pdf) explains in more
details the early stages of the work (not up to date).

The results and training procedure can also be visualized in
[milestone-1/example.ipynb](milestone-1/example.ipynb).
