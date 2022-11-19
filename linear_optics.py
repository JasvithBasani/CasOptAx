{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPuXBs99JmrmPAz7fblCH8E"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "#Jasvith Raj Basani\n",
        "#\n",
        "#\n",
        "#History\n",
        "# 15/11/2022 - Created this File"
      ],
      "metadata": {
        "id": "yIOXWlkUUhG8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 79,
      "metadata": {
        "id": "IlntMsQYUZ1Q"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import jax.numpy as jnp\n",
        "from jax import jit\n",
        "import scipy.stats\n",
        "from functools import partial"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class Linear_Optics:\n",
        "\n",
        "  def __init__(self,\n",
        "               N_modes: jnp.int16 = None,\n",
        "               ):\n",
        "    \n",
        "    r\"\"\"\n",
        "    Class of differentiable linear optics tools.\n",
        "    In particular, returns matrices implemented by photonic integrated meshes\n",
        "    param N_modes: Number of spatial modes input to the network.\n",
        "    \"\"\"\n",
        "    self.N_modes = N_modes\n",
        "\n",
        "  def haar_mat(self, N_modes):\n",
        "    r\"\"\"\n",
        "    Returns NxN Haar random unitary matrix\n",
        "    param N_modes: Number of spatial modes input to the network.\n",
        "    \"\"\"\n",
        "    return scipy.stats.unitary_group.rvs(N_modes)\n",
        "\n",
        "\n",
        "  @partial(jit, static_argnums = (0, ))\n",
        "  def MZI(self, theta, phi, alpha = 0 + 0j, beta = 0 + 0j):\n",
        "    assert self.N_modes == 2\n",
        "\n",
        "    r\"\"\"\n",
        "    Single MZI transfer function, to return a 2x2 unitary transformation\n",
        "    param theta: Phase shift value\n",
        "    param phi: Phase shift values\n",
        "    param alpha: Beamsplitter error value\n",
        "    param beta: Beamsplitter error value\n",
        "    Matrix given by: eq (5) in arXiv:2103.04993\n",
        "    \"\"\"\n",
        "\n",
        "    t_00 = jnp.exp(1j * phi) * (jnp.cos(alpha - beta) * jnp.sin(theta/2) + 1j * jnp.sin(alpha + beta) * jnp.cos(theta/2))\n",
        "    t_01 = (jnp.cos(alpha + beta) * jnp.cos(theta/2) + 1j * jnp.sin(alpha - beta) * jnp.sin(theta/2))\n",
        "    t_10 = jnp.exp(1j * phi) * (jnp.cos(alpha + beta) * jnp.cos(theta/2) - 1j * jnp.sin(alpha - beta) * jnp.sin(theta/2))\n",
        "    t_11 = -(jnp.cos(alpha - beta) * jnp.sin(theta/2) + 1j * jnp.sin(alpha + beta) * jnp.cos(theta/2))\n",
        "    T = 1j * jnp.exp(1j * theta/2) * jnp.array([[t_00, t_01],\n",
        "                                                [t_10, t_11]])\n",
        "    return T\n",
        "\n",
        "\n",
        "  def clements(theta, phi, D, alpha, beta):\n",
        "\n",
        "    r\"\"\"\n",
        "    Differentiable clements matrix, to return a NxN unitary transformation\n",
        "    param theta: 1D array of phase shift values - MZIs are indexed from top to bottom and left to right\n",
        "    param phi: 1D array of phase shift values\n",
        "    param D: 1D array for output phase screen\n",
        "    param alpha: 1D array for directional coupler errors\n",
        "    param beta: 1D array for directional coupler errors\n",
        "    \"\"\"\n",
        "\n",
        "    raise NotImplementedError() "
      ],
      "metadata": {
        "id": "qvjJhHwpU8et"
      },
      "execution_count": 80,
      "outputs": []
    }
  ]
}