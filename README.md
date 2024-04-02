# Baysian models for ODE models in SBML 

This project implements Bayesian models using [PyMC](https://www.pymc.io) on top of ODE-based models encoded in the [Systems Biology Markup Language](https://sbml.org/) (SBML).

## Motivation

[Systems Biology Markup Language](https://sbml.org/) (SBML) provides an intuitive and reproducible way to define ordinary differential equation (ODE) models in systems biology and systems medicine. Here we outline an attempt to build a Bayesian framework to quantify the uncertainty of estimates associated with physiologically based pharmacokinetic (PBPK) models encoded in SBML.


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


# Example

In this particular case, [a simple two compartment model](src/parameter_variability/models/sbml/simple_pk.md)


To generate the toy example, the two compartment model is fed draws from an idealized random distribution for each parameter. They are called the `true_thetas`. 
Then a forward simulation is performed to generate a run simulation for each theta. 

After adding noise to the simulation(s), a Bayesian model fits the data and draw sample from a posterior distribution. 
The empirical distribution of those samples should contain the `true_thetas`.

## Example ODEs
```


```


## Current modelled parameters
- `k`: Absorption constant
- `CL` Clearance constant


## Running the model

After the setup below, simply run the script `bayesian_example.py` on your favourite IDE or run it in the command line from the main directory as follows

```bash
(parameter-variability) python src/parameter_variability/bayes/bayes_example.py
```

## Example outputs

Plots of results for the analysis on the Gut compartment

*Figure 1*: Sampling random parameters from "true" distribution

<img src="img/01-parameter_sampling.png" alt="01-parameter_sampling" width="200"/>

*Figure 2*: Toy Data simulated using values from the true distribution

<img src="img/02-simulation_plotting.png" alt="02-simulation_plotting" width="200"/>

*Figure 3*: Graph representing the Bayesian Model

<img src="img/03-bayesian_model.png" alt="03-bayesian_model" width="200"/>

*Figure 4*: Trace Plot of the parameters sampled from the Bayesian model

<img src="img/04-trace_plot.png" alt="04-trace_plot" width="200"/>

*Figure 5*: Proposed simulations sampled from the Bayesian Model

<img src="img/05-bayesian_sample.png" alt="05-bayesian_sample" width="200"/>


# License

* Source Code: [LGPLv3](http://opensource.org/licenses/LGPL-3.0)
* Documentation: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

The parameter-variability source is released under both the GPL and LGPL licenses version 2 or later. You may choose which license you choose to use the software under.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License or the GNU Lesser General Public
License as published by the Free Software Foundation, either version 2 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.


# Funding

Matthias König is supported by the Federal Ministry of Education and Research (BMBF, Germany) within the research network Systems Medicine of the Liver (**LiSyM**, grant number 031L0054) and by the German Research Foundation (DFG) within the Research Unit Programme FOR 5151 [QuaLiPerF](https://qualiperf.de) (Quantifying Liver Perfusion-Function Relationship in Complex Resection - A Systems Medicine Approach)" by grant number 436883643 and by grant number 465194077 (Priority Programme SPP 2311, Subproject SimLivA).

© 2023-2024 Antonio Alvarez and [Matthias König](https://livermetabolism.com)
