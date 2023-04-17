# Reinforcement Learning in Fluid Simulations with the LBM Method

**This is still work in progress...**

This project aims to provide reinforcement learning environments for controlling various aspects of CFD simulations. The goal is to test various control tasks for RL agents in CFD context, such as controlling a boundary condition, performing live data assimilation, or interacting with the solver core model to make, for example, a turbulence model. The CFD code is written in JAX for optimal performance.

## Introduction
CFD simulations can be challenging to control due to their complex nature. This project aims to provide RL environments that allow users to train agents to control different aspects of the simulation. The ultimate goal is to develop agents that can control CFD simulations more efficiently than traditional methods, such as manually tuning parameters.
The core CFD simulator is written in JAX, which provides just-in-time compilation as well as full GPU support. It can also be used to back-propagate gradients through the simulation solver with autodifferentiation. This can be interesting for optimizing control tasks, though I did not play with it yet.

## Getting Started
To get started with this project, follow these steps:

Clone the repository to your local machine and install the package.
```shell
git clone https://github.com/hlasco/rllbm && cd rllbm
pip install .
```

## Usage

The [examples](examples) folder contains sample codes to illustrate how to use this package:
- [examples/cfd-rayleigh_benard](examples/cfd-rayleigh_benard) and [examples/cfd-von_karman](examples/cfd-von_karman) shows how to use the LBM simulation code on canonical problems (e.g. Von Karman vortex street, Rayleigh-Benard convection).
- [examples/rl-heat_wall](examples/rl-heat_wall) shows how to create a RL environment and train an agent with the Ray-Rllib library. In this example, the agent controls the boundary temperature of a closed domain, which triggers convective motions. The agent's goal is to drive a passive fluid particle towards a target location.

**TODO**
### Defining a LBM simulation
#### `Simulation`
#### `Lattice`
#### `Stencil`
#### The `stream` and `collide` function
#### `CoupledLattices`
#### `Boundary` and `BoundaryDict`
### `LBMEnv`

