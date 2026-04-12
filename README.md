# Cascaded Optical Systems Approach to Neural Networks (CasOptAx)

![](/assets/qonn_schematic.png)


## Overview

This repository contains code to model quantum photonic neural networks using linear optical elements and cavity QED nonlinearities. The detailed architecture is introduced in the manuscript "[Universal Logical Quantum Photonic Neural Network Processor via Cavity-Assisted Interactions](https://arxiv.org/abs/2410.02088)" by Basani, Niu, and Waks(2024). Linear optical transformations over time-binned modes using the generalized Green Machine is introduced in the manuscript "[Hardware-Efficient Large-Scale Universal Linear Transformations for Optical Modes in the Synthetic Time Dimension](https://arxiv.org/abs/2505.00865)" by Basani*, Cui*, Postlewaite, Waks and Guha(2025).

## Components

-`linear_optics.py`: helper functions for linear optical unitary evolution
-`scatterer.py`: scattering matrix generator for multimode 2LS nonlinearity
-`circuit_builder.py`: main models to to simulate forward scattering with both single-mode and multimode circuits 
-`utils.py`: miscellaneous utilities
-`meshes_utils.py`: helper functions to make this repository compatible with the meshes package
-`spin_network.py`: ~
-`Tutorials`: 
    `QPNN_Haar_Random_Tutorial.ipynb`: Tutorial file to optimize the phases of a 2 layer QPNN to generate a 3-photon Haar-random state
    `Boosted_BSM_tutorial.ipynb`: Tutorial file to calculate the success and error rates of the boosted Bell State Measurement


Training histories are written to `npy` files which are not included in this repository, but are available upon request from the authors.

## Citing

If you found this useful, please cite us using: 

```
@article{basani2025universal,
  title={Universal logical quantum photonic neural network processor via cavity-assisted interactions},
  author={Basani, Jasvith Raj and Niu, Murphy Yuezhen and Waks, Edo},
  journal={npj Quantum Information},
  volume={11},
  number={1},
  pages={142},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
```
@article{basani2026hardware,
  title={Hardware-Efficient Universal Linear Transformations for Optical Modes in the Synthetic Time Dimension},
  author={Basani, Jasvith Raj and Cui, Chaohan and Postlewaite, Jack and Waks, Edo and Guha, Saikat},
  journal={PRX Quantum},
  volume={7},
  number={2},
  pages={020305},
  year={2026},
  publisher={APS}
}
```
