# Reinforcement Learning in FLuid Simulations with the LBM Method
This project provides reinforcement learning environments for controlling various aspects of CFD simulations. The goal is to test various control tasks for RL agents in CFD context, such as controlling a boundary condition, performing live data assimilation, or interacting with the solver core model to make, for example, a turbulence model. The CFD code is written in JAX for optimal performance.

## Introduction
CFD simulations can be challenging to control due to their complex nature. This project aims to provide RL environments that allow users to train agents to control different aspects of the simulation. The ultimate goal is to develop agents that can control CFD simulations more efficiently than traditional methods, such as manually tuning parameters.

## Getting Started
To get started with this project, follow these steps:

Clone the repository to your local machine and install the package.
```shell
git clone https://github.com/hlasco/rllbm && cd rllbm
pip install .
```

## Usage
The `example` folder contains sample codes to illustrate how to use this package:
- `example/lbm` shows how to use the LBM simulation code on canonical problems (e.g. Von Karman vortex street, Rayleigh-Benard convection).
- (Work in progress) `example/env` shows how to create RL environments for training agents.
- (Work in progress) `example/agent` shows how to train RL agents using the environments.
### LBM Simulations
#### Lattice model
#### Coupled lattices
#### Initial Conditions
#### Boundary conditions
#### Visualization
### Environments
### Agents
