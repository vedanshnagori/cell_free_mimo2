# rates.py
"""
Compute user rates (download speeds) in the cell-free MIMO system.

Key formula (Eq. 5 from paper):
    R_k = log2(1 + signal / (interference + noise))

where:
    signal       = sum of |h · v|² from APs assigned to user k
    interference  = sum of |h · v|² from APs NOT assigned to user k  
    noise        = 1 (normalized)
"""

import numpy as np
import config


def compute_mr_precoding(H):
    """
    Maximum Ratio precoding: point the beam toward the user.
    
    v_{m,k} = conj(h_{m,k}) / ||h_{m,k}||
    
    This maximizes received signal power (as we just learned!).
    
    Args:
        H: channel dict, H[(m,k)] = complex vector of shape (N_TX,)
    
    Returns:
        V_MR: dict, V_MR[(m,k)] = precoding vector of shape (N_TX,)
    """
    V_MR = {}
    
    for m in range(config.N_AP):
        for k in range(config.N_USER):
            h = H[(m, k)]
            h_conj = np.conj(h)
            norm = np.linalg.norm(h_conj)
            
            if norm > 1e-10:
                V_MR[(m, k)] = h_conj / norm
            else:
                # If channel is basically zero, use equal power
                V_MR[(m, k)] = np.ones(config.N_TX, dtype=complex) / np.sqrt(config.N_TX)
    
    return V_MR


def compute_user_rates(H, gamma, V):
    """
    Compute download rate for each user.
    
    For user k:
        signal = sum over APs assigned to k: rho * |h_{m,k}^T · v_m|²
        interference = sum over APs NOT assigned to k: mu * rho * |h_{n,k}^T · v_n|²
        R_k = log2(1 + signal / (interference + 1))
    
    Args:
        H: channel dict, H[(m,k)] = complex vector
        gamma: (N_AP, N_USER) binary assignment matrix
               gamma[m,k] = 1 means AP m serves user k
        V: dict, V[m] = precoding vector for AP m
    
    Returns:
        rates: (N_USER,) array of rates in bits/s/Hz
    """
    rates = np.zeros(config.N_USER)
    
    for k in range(config.N_USER):
        signal = 0.0
        interference = 0.0
        
        for m in range(config.N_AP):
            h_mk = H[(m, k)]       # Channel from AP m to user k
            v_m = V[m]              # How AP m is transmitting
            
            # Beamforming gain: how much signal user k receives from AP m
            # This is |h^T · v|² — the dot product squared
            gain = np.abs(np.dot(np.conj(h_mk), v_m)) ** 2
            
            if gamma[m, k] == 1:
                # AP m is assigned to user k → this is DESIRED signal
                signal += config.RHO * gain
            else:
                # AP m is assigned to someone else → this is INTERFERENCE
                interference += config.MU_NK * config.RHO * gain
        
        # Shannon formula
        rates[k] = np.log2(1 + signal / (interference + 1))
    
    return rates


def compute_min_rate(H, gamma, V):
    """
    The objective: minimum rate among all users.
    We want to MAXIMIZE this (max-min fairness).
    """
    rates = compute_user_rates(H, gamma, V)
    return np.min(rates)


def compute_sum_rate(H, gamma, V):
    """Total rate across all users."""
    rates = compute_user_rates(H, gamma, V)
    return np.sum(rates)


def build_precoding_from_assignment(H, gamma):
    """
    Given an assignment gamma, build MR precoding for each AP.
    
    Each AP uses MR precoding toward its assigned user.
    
    Args:
        H: channel dict
        gamma: (N_AP, N_USER) assignment matrix
    
    Returns:
        V: dict, V[m] = precoding vector for AP m
    """
    V_MR = compute_mr_precoding(H)
    V = {}
    
    for m in range(config.N_AP):
        assigned_users = np.where(gamma[m] == 1)[0]
        
        if len(assigned_users) > 0:
            # Use MR precoding toward the assigned user
            k = assigned_users[0]
            V[m] = V_MR[(m, k)]
        else:
            # No user assigned — use zero precoding
            V[m] = np.zeros(config.N_TX, dtype=complex)
    
    return V


# ---- Quick test ----
if __name__ == "__main__":
    from channel import generate_positions, compute_distances, generate_sample
    
    # Setup
    ap_pos, user_pos = generate_positions(seed=42)
    dist = compute_distances(ap_pos, user_pos)
    H, features = generate_sample(dist)
    
    print("=" * 50)
    print("Testing rate computation")
    print("=" * 50)
    
    # Show channel strengths
    print("\nChannel strengths |h| for each AP-User pair:")
    print(f"{'':>8}", end="")
    for k in range(config.N_USER):
        print(f"  User{k}", end="")
    print()
    
    for m in range(config.N_AP):
        print(f"  AP{m}:  ", end="")
        for k in range(config.N_USER):
            strength = np.linalg.norm(H[(m, k)])
            print(f"  {strength:.3f}", end="")
        print()
    
    # Test assignment 1: Round-robin (AP0→User0, AP1→User1, etc.)
    gamma1 = np.zeros((config.N_AP, config.N_USER))
    for m in range(config.N_AP):
        gamma1[m, m % config.N_USER] = 1
    
    V1 = build_precoding_from_assignment(H, gamma1)
    rates1 = compute_user_rates(H, gamma1, V1)
    
    print("\n--- Assignment 1: Round-robin ---")
    print(f"  Assignment:\n{gamma1.astype(int)}")
    print(f"  User rates: {np.round(rates1, 4)}")
    print(f"  Min rate:   {np.min(rates1):.4f}")
    print(f"  Sum rate:   {np.sum(rates1):.4f}")
    
    # Test assignment 2: All APs serve User 0 (bad for other users!)
    gamma2 = np.zeros((config.N_AP, config.N_USER))
    gamma2[:, 0] = 1  # All APs assigned to User 0
    
    V2 = build_precoding_from_assignment(H, gamma2)
    rates2 = compute_user_rates(H, gamma2, V2)
    
    print("\n--- Assignment 2: All APs → User 0 ---")
    print(f"  Assignment:\n{gamma2.astype(int)}")
    print(f"  User rates: {np.round(rates2, 4)}")
    print(f"  Min rate:   {np.min(rates2):.4f}  ← terrible! Users 1,2 get nothing")
    print(f"  Sum rate:   {np.sum(rates2):.4f}")
    
    # Test assignment 3: Based on strongest channels
    gamma3 = np.zeros((config.N_AP, config.N_USER))
    # Assign each AP to the user with the strongest channel
    for m in range(config.N_AP):
        strengths = [np.linalg.norm(H[(m, k)]) for k in range(config.N_USER)]
        best_user = np.argmax(strengths)
        gamma3[m, best_user] = 1
    
    # Make sure each user has at least one AP
    for k in range(config.N_USER):
        if np.sum(gamma3[:, k]) == 0:
            # Find AP with weakest current assignment
            m = np.random.randint(0, config.N_AP)
            gamma3[m, :] = 0
            gamma3[m, k] = 1
    
    V3 = build_precoding_from_assignment(H, gamma3)
    rates3 = compute_user_rates(H, gamma3, V3)
    
    print("\n--- Assignment 3: Strongest channel ---")
    print(f"  Assignment:\n{gamma3.astype(int)}")
    print(f"  User rates: {np.round(rates3, 4)}")
    print(f"  Min rate:   {np.min(rates3):.4f}")
    print(f"  Sum rate:   {np.sum(rates3):.4f}")
    
    print("\n" + "=" * 50)
    print("KEY INSIGHT: Different assignments give different rates!")
    print("The QNN's job is to LEARN the best assignment.")
    print("=" * 50)