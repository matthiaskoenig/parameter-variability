# parameter-variability

First exploration of fitting Bayesian Models on top of a SBML-based model using a toy example

## Motivation

[Systems Biology Markup Language](https://sbml.org/) (SBML) represents an intuitive way to define ordinary differential equation models in application in Biology, Chemistry and [many more](https://sbml.org/about/contributors/). 
Here we outline an attempt to build a Bayesian framework to quantify the uncertainty of estimates pertaining to Physiologically Based Pharmacokinetic (PBPK) models.

In this particular case, [a simple two compartment model](src/parameter_variability/models/sbml/simple_pk.md)

## Description

To generate the toy example, the two compartment model is fed draws from an idealized random distribution for each parameter. They are called the `true_thetas`. 
Then a forward simulation is performed to generate a run simulation for each theta. 

After adding noise to the simulation(s), a Bayesian model fits the data and draw sample from a posterior distribution. 
The empirical distribution of those samples should contain the `true_thetas`.

## Current modelled parameters
- `k`: Absorption constant
- `CL` Clearance constant


## Installation

Create a virtual environment and install the dependencies

### libraries
```bash
sudo apt-get -y install graphviz graphviz-dev
```


### virtual environment
```bash
mkvirtualenv parameter-variability --python=python3.10
(parameter-variability) pip install -r requirements.txt
```

## Running the model

After the setup below, simply run the script `bayesian_example.py` on your favourite IDE or run it in the command line from the main directory as follows

```bash
(parameter-variability) python src/parameter_variability/bayes/bayes_example.py
```
