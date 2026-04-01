# config.py
"""
All system parameters from Table III of the paper.
One place to change everything.
"""

import numpy as np


# --- System Model (the wireless network) ---
N_AP = 4            # Number of Access Points (routers)
N_USER = 3          # Number of users (phones)
N_TX = 2            # Number of antennas per AP
N_PATH = 3          # Number of signal paths (reflections)
KAPPA = 2.3         # How fast signal weakens with distance
SIGMA2 = 1.0        # Background noise power
P_T = 1.0           # Transmit power per AP
RHO = P_T / SIGMA2  # Signal-to-noise ratio
MU_NK = 0.1         # Interference factor between APs

# --- Training ---
N_DATA = 100        # Training samples
N_EPOCH = 100       # Training epochs
LR = 0.01           # Learning rate
R_PENALTY = -10     # Penalty when a user gets no AP

# --- QNN Architecture ---
N_LAYER_CLOUD = 3   # Layers in cloud QNN
N_LAYER_EDGE = 3    # Layers in edge QNN

# --- Reproducibility ---
SEED = 42