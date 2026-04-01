# baselines.py
"""
Baseline methods for comparison.

1. Search-based: greedy assignment (Algorithm 4, Appendix C)
2. Random: random assignment (lower bound)

These show how the QNN compares to classical methods.
"""

import numpy as np
import config
from rates import compute_mr_precoding, compute_user_rates


def search_based_assignment(H):
    """
    Greedy search for best assignment (Algorithm 4).
    
    For each user, find the available AP with highest SINR.
    This is a reasonable classical baseline.
    """
    V_MR = compute_mr_precoding(H)
    gamma = np.zeros((config.N_AP, config.N_USER))
    used_aps = set()
    
    # For each user, find the best available AP
    for k in range(config.N_USER):
        best_sinr = -np.inf
        best_ap = -1
        
        for m in range(config.N_AP):
            if m in used_aps:
                continue
            
            v_m = V_MR[(m, k)]
            h_mk = H[(m, k)]
            
            # Signal strength
            signal = config.RHO * np.abs(np.dot(np.conj(h_mk), v_m)) ** 2
            
            # Interference from already-assigned APs
            interference = 0.0
            for n in used_aps:
                assigned_user = np.argmax(gamma[n])
                v_n = V_MR[(n, assigned_user)]
                h_nk = H[(n, k)]
                interference += config.MU_NK * config.RHO * np.abs(
                    np.dot(np.conj(h_nk), v_n)) ** 2
            
            sinr = signal / (interference + 1)
            
            if sinr > best_sinr:
                best_sinr = sinr
                best_ap = m
        
        if best_ap >= 0:
            gamma[best_ap, k] = 1
            used_aps.add(best_ap)
    
    # Assign remaining APs to weakest user
    for m in range(config.N_AP):
        if m not in used_aps:
            V_temp = {}
            for n in range(config.N_AP):
                assigned = np.where(gamma[n] == 1)[0]
                if len(assigned) > 0:
                    V_temp[n] = V_MR[(n, assigned[0])]
                else:
                    V_temp[n] = np.zeros(config.N_TX, dtype=complex)
            
            rates = compute_user_rates(H, gamma, V_temp)
            weakest = np.argmin(rates)
            gamma[m, weakest] = 1
            used_aps.add(m)
    
    return gamma


def random_assignment(H):
    """
    Random AP-user assignment (lower bound).
    Each AP randomly picks a user. Ensures coverage.
    """
    gamma = np.zeros((config.N_AP, config.N_USER))
    
    for m in range(config.N_AP):
        k = np.random.randint(0, config.N_USER)
        gamma[m, k] = 1
    
    # Ensure each user has at least one AP
    for k in range(config.N_USER):
        if np.sum(gamma[:, k]) == 0:
            # Steal from user with most APs
            user_counts = np.sum(gamma, axis=0)
            rich_user = np.argmax(user_counts)
            rich_aps = np.where(gamma[:, rich_user] == 1)[0]
            if len(rich_aps) > 1:
                gamma[rich_aps[0], rich_user] = 0
                gamma[rich_aps[0], k] = 1
            else:
                m = np.random.randint(0, config.N_AP)
                gamma[m, :] = 0
                gamma[m, k] = 1
    
    return gamma


def evaluate_method(dataset, assignment_fn):
    """
    Evaluate an assignment method on a dataset.
    
    Returns average min rate and sum rate.
    """
    min_rates = []
    sum_rates = []
    
    for sample in dataset:
        H = sample['H']
        gamma = assignment_fn(H)
        
        # Use MR precoding
        V_MR = compute_mr_precoding(H)
        V = {}
        for m in range(config.N_AP):
            assigned = np.where(gamma[m] == 1)[0]
            if len(assigned) > 0:
                V[m] = V_MR[(m, assigned[0])]
            else:
                V[m] = np.zeros(config.N_TX, dtype=complex)
        
        rates = compute_user_rates(H, gamma, V)
        min_rates.append(np.min(rates))
        sum_rates.append(np.sum(rates))
    
    return {
        'avg_min_rate': np.mean(min_rates),
        'avg_sum_rate': np.mean(sum_rates),
        'std_min_rate': np.std(min_rates),
        'std_sum_rate': np.std(sum_rates),
    }


if __name__ == "__main__":
    from channel import generate_positions, compute_distances, generate_sample
    from train import generate_dataset
    
    ap_pos, user_pos = generate_positions(seed=42)
    dist = compute_distances(ap_pos, user_pos)
    dataset = generate_dataset(50, dist, seed=42)
    
    print("Evaluating baselines on 50 samples...")
    
    search_res = evaluate_method(dataset, search_based_assignment)
    random_res = evaluate_method(dataset, random_assignment)
    
    print(f"\nSearch-based: Min Rate={search_res['avg_min_rate']:.4f}, "
          f"Sum Rate={search_res['avg_sum_rate']:.4f}")
    print(f"Random:       Min Rate={random_res['avg_min_rate']:.4f}, "
          f"Sum Rate={random_res['avg_sum_rate']:.4f}")