# model: simple_chain
Autogenerated ODE System from SBML file with sbmlutils.
```
time: [-]
substance: [-]
extent: [-]
volume: [-]
area: [-]
length: [-]
```

## Parameters `p`
```
k1 = 1.0  # [-] 
liver = 1.0  # [-] 
```

## Initial conditions `x0`
```
S1 = 1.0  # [-/-] liver
S2 = 0.0  # [-/-] liver
```

## ODE system
```
# y
R1 = k1 * S1  # [-/-]

# odes
d S1/dt = -R1 / liver  # [-/-/-]
d S2/dt = R1 / liver  # [-/-/-]
```