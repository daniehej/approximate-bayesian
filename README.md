# approximate-bayesian
Code for the project: Calibration of Radio Channel Model using Approximate Bayesian Computation.

D.G. Andersen, L. Menholt, S.B.B. Thomsen, D.B. van Diepen.

## Dependencies
```
numpy
matplotlib
scipy
```

## Files
`ABC.py` - Apply approximate Bayesian computation to estimate radio channel model parameters.

`Spencer_model.py` - Implementation of the Spencer radio cahnnel model for use in simulations.

`SV_model.py` - Implementation of the simpler Saleh-Valenzuela channel model.

`Summary_statistics.py` - Experiments with different summary statistics for use in the estimation.

## Usage
The main script is run in order to calibrate the channel model, including generating figures to show the performance.
```
python ABC.py
```

The other scripts can also be run for the purpose of testing and generating additional figures.
