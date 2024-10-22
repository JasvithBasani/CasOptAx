# Cascaded Optical Systems Approach to Neural Networks (CasOptAx)

![](/assets/qonn_schematic.png)


## Overview

This repository contains code to model quantum photonic neural networks using linear optical elements and cavity QED nonlinearities. The detailed architecture is introduced in the manuscript "[Universal Logical Quantum Photonic Neural Network Processor via Cavity-Assisted Interactions](https://arxiv.org/abs/2410.02088)" by Basani, Niu, and Waks(2024).

## Components

-`linear_optics.py`: helper functions for linear optical unitary evolution
-`scatterer.py`: scattering matrix generator for multimode 2LS nonlinearity
-`circuit_builder.py`: main models to to simulate forward scattering with both single-mode and multimode circuits 
-`utils.py`: miscellaneous utilities
-`meshes_utils.py`: helper functions to make this repository compatible with the meshes package
-`spin_network.py`: ~
-`Tutorials`: 
    `QPNN_Haar_Random_Tutorial.ipynb`: Tutorial file to optimize the phases of a 2 layer QPNN to generate a 3-photon Haar-random state


Training histories are written to `npy` files which are not included in this repository, but are available upon request from the authors.

## Citing

If you found this useful, please cite us using: 

```
@article{basani2024universal,
  title={Universal Logical Quantum Photonic Neural Network Processor via Cavity-Assisted Interactions},
  author={Basani, Jasvith Raj and Niu, Murphy Yuezhen and Waks, Edo},
  journal={arXiv preprint arXiv:2410.02088},
  year={2024}
}
```
