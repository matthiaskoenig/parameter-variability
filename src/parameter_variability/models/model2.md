# model: model2
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
CL = 1.0  # [-] 
Q = 1.0  # [-] 
Vcent = 1.0  # [-] 
Vgut = 1.0  # [-] 
Vperi = 1.0  # [-] 
k = 1.0  # [-] 
```

## Initial conditions `x0`
```
y_cent = 0.0  # [-/-] Vcent
y_gut = 1.0  # [-/-] Vgut
y_peri = 0.0  # [-/-] Vperi
```

## ODE system
```
# y
ABSORPTION = -k * y_gut  # [-/-]
CLEARANCE = -CL * y_cent  # [-/-]
R1 = Q * y_cent  # [-/-]
R2 = Q * y_peri  # [-/-]

# odes
d y_cent/dt = (ABSORPTION / Vcent - CLEARANCE / Vcent - R1 / Vcent) + R2 / Vcent  # [-/-/-]
d y_gut/dt = -ABSORPTION / Vgut  # [-/-/-]
d y_peri/dt = R1 / Vperi - R2 / Vperi  # [-/-/-]
```