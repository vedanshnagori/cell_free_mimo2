# channel.py
"""
Cell-Free MIMO Channel Model.
Generates the wireless channels between APs and users.

Key concept: h_{m,k} is a complex vector telling us how the signal
changes between AP m and user k. Bigger |h| = better connection.
"""

import numpy as np
import config


def generate_positions(seed=None):
    """
    Place APs and users randomly in a 1×1 square area.
    
    Returns:
        ap_pos:   (N_AP, 2) array - x,y positions of APs
        user_pos: (N_USER, 2) array - x,y positions of users
    """
    if seed is not None:
        np.random.seed(seed)
    
    ap_pos = np.random.uniform(0, 1, size=(config.N_AP, 2))
    user_pos = np.random.uniform(0, 1, size=(config.N_USER, 2))
    
    return ap_pos, user_pos


def compute_distances(ap_pos, user_pos):
    """
    Compute distance between every AP-user pair.
    
    Returns:
        distances: (N_AP, N_USER) matrix
                   distances[m, k] = distance between AP m and user k
                   Normalized to [0, 1] range
    """
    N_AP = len(ap_pos)
    N_USER = len(user_pos)
    distances = np.zeros((N_AP, N_USER))
    
    for m in range(N_AP):
        for k in range(N_USER):
            # Euclidean distance
            diff = ap_pos[m] - user_pos[k]
            distances[m, k] = np.sqrt(diff[0]**2 + diff[1]**2)
    
    # Normalize so max distance = 1
    max_dist = np.max(distances)
    if max_dist > 0:
        distances = distances / max_dist
    
    # Avoid zero distance (would cause division by zero)
    distances = np.maximum(distances, 0.01)
    
    return distances


def generate_channel(distances):
    """
    Generate random channel vectors for all AP-user pairs.
    
    For each AP m and user k:
        h_{m,k} is a complex vector of size N_TX
    
    Closer AP-user pairs get stronger channels (bigger |h|).
    
    The formula (simplified from Eq. 3):
        h_{m,k}[j] = (1/sqrt(N_PATH)) * sum of random_complex * distance_factor
    
    Returns:
        H: dict where H[(m, k)] = complex numpy array of shape (N_TX,)
    """
    H = {}
    
    for m in range(config.N_AP):
        for k in range(config.N_USER):
            # Distance-based signal strength
            # Farther away = weaker signal (this is "pathloss")
            channel_strength = distances[m, k] ** (-config.KAPPA)
            
            # Generate the channel vector (one element per antenna)
            h_mk = np.zeros(config.N_TX, dtype=complex)
            
            for j in range(config.N_TX):
                # Sum over multiple signal paths (reflections off walls, etc.)
                for n in range(config.N_PATH):
                    # Random complex number (Rayleigh fading)
                    # This models the random nature of wireless
                    random_gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                    
                    # Scale by distance-based strength
                    random_gain *= np.sqrt(channel_strength)
                    
                    # Random phase from antenna array
                    angle = np.random.uniform(0, 2 * np.pi)
                    z = j - 0.5 * (config.N_TX - 1)
                    steering = np.exp(-1j * 2 * np.pi * angle * z)
                    
                    h_mk[j] += random_gain * steering
                
                # Average over paths
                h_mk[j] /= np.sqrt(config.N_PATH)
            
            H[(m, k)] = h_mk
    
    return H


def channel_to_features(H):
    """
    Convert complex channel dict into a real-valued matrix
    that can be fed into a QNN.
    
    Complex numbers have real and imaginary parts:
        h = a + bj  →  we store [a, b]
    
    Returns:
        features: (N_AP, N_USER * 2 * N_TX) real-valued matrix
    """
    rows = []
    for m in range(config.N_AP):
        row = []
        for k in range(config.N_USER):
            h = H[(m, k)]
            # Split complex into real and imaginary parts
            row.extend(np.real(h).tolist())
            row.extend(np.imag(h).tolist())
        rows.append(row)
    
    return np.array(rows)


def generate_sample(distances):
    """
    Generate one complete channel sample.
    
    Returns:
        H: channel dict
        features: real-valued feature matrix for QNN input
    """
    H = generate_channel(distances)
    features = channel_to_features(H)
    return H, features


# ---- Quick test ----
if __name__ == "__main__":
    # Generate positions
    ap_pos, user_pos = generate_positions(seed=42)
    print("AP positions:")
    print(ap_pos)
    print("\nUser positions:")
    print(user_pos)
    
    # Compute distances
    dist = compute_distances(ap_pos, user_pos)
    print("\nDistances (AP x User):")
    print(np.round(dist, 3))
    
    # Generate channel
    H, features = generate_sample(dist)
    
    print("\nChannel h_(AP0, User0):", H[(0, 0)])
    print("  Strength |h|:", np.round(np.abs(H[(0, 0)]), 3))
    
    print("\nChannel h_(AP0, User1):", H[(0, 1)])
    print("  Strength |h|:", np.round(np.abs(H[(0, 1)]), 3))
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"  ({config.N_AP} APs × {config.N_USER * 2 * config.N_TX} features per AP)")