# main.py
"""
Main entry point.
Runs the full experiment: train QNN, evaluate, compare, plot.
"""

from pennylane import numpy as np
import config
from channel import generate_positions, compute_distances, generate_sample
from rates import compute_mr_precoding, compute_user_rates
from train import train, generate_dataset
from baselines import (search_based_assignment, random_assignment,
                       evaluate_method)
from cloud_qnn import CloudQNN
from edge_qnn import EdgeQNN
from plot import plot_training_curves, plot_comparison, plot_network


def evaluate_qnn(cloud, edges, dataset):
    """Evaluate trained QNN on a dataset."""
    min_rates = []
    sum_rates = []
    
    for sample in dataset:
        H = sample['H']
        H_matrix = sample['features']
        
        # Cloud QNN → assignment
        features = cloud.prepare_features(H_matrix)
        output = cloud.forward(features)
        gamma = cloud.decode_assignment(output)
        
        # Edge QNNs → precoding
        V_MR = compute_mr_precoding(H)
        V = {}
        for m in range(config.N_AP):
            assigned = np.where(gamma[m] == 1)[0]
            if len(assigned) > 0:
                edge_feat = edges[m].prepare_features(H)
                edge_out = edges[m].forward(edge_feat)
                V[m] = edges[m].decode_precoding(edge_out)
            else:
                V[m] = np.zeros(config.N_TX, dtype=complex)
        
        rates = compute_user_rates(H, gamma, V)
        min_rates.append(float(np.min(rates)))
        sum_rates.append(float(np.sum(rates)))
    
    return {
        'avg_min_rate': np.mean(min_rates),
        'avg_sum_rate': np.mean(sum_rates),
        'std_min_rate': np.std(min_rates),
        'std_sum_rate': np.std(sum_rates),
    }


def main():
    config_info = (f"N_AP={config.N_AP}, N_USER={config.N_USER}, "
                   f"N_TX={config.N_TX}, N_DATA={config.N_DATA}, "
                   f"N_EPOCH={config.N_EPOCH}")
    
    print("=" * 60)
    print("NON-CENTRALIZED QNN FOR CELL-FREE MIMO")
    print("=" * 60)
    print(f"  Config: {config_info}\n")
    
    # ---- Step 1: Train ----
    print("[1/5] Training QNN...")
    cloud, edges, history, distances, ap_pos, user_pos = train()
    
    # ---- Step 2: Generate test data ----
    print("\n[2/5] Generating test data...")
    test_data = generate_dataset(50, distances, seed=config.SEED + 999)
    print(f"  Test samples: {len(test_data)}")
    
    # ---- Step 3: Evaluate all methods ----
    print("\n[3/5] Evaluating methods...")
    
    qnn_res = evaluate_qnn(cloud, edges, test_data)
    search_res = evaluate_method(test_data, search_based_assignment)
    random_res = evaluate_method(test_data, random_assignment)
    
    print(f"\n  {'Method':<20} {'Min Rate':>10} {'Sum Rate':>10}")
    print(f"  {'-'*40}")
    print(f"  {'QNN (proposed)':<20} {qnn_res['avg_min_rate']:>10.4f} "
          f"{qnn_res['avg_sum_rate']:>10.4f}")
    print(f"  {'Search (baseline)':<20} {search_res['avg_min_rate']:>10.4f} "
          f"{search_res['avg_sum_rate']:>10.4f}")
    print(f"  {'Random':<20} {random_res['avg_min_rate']:>10.4f} "
          f"{random_res['avg_sum_rate']:>10.4f}")
    
    # ---- Step 4: Plot results ----
    print("\n[4/5] Generating plots...")
    plot_training_curves(history)
    plot_comparison(qnn_res, search_res, random_res)
    plot_network(ap_pos, user_pos)
    
    # Plot one example with assignment
    sample = test_data[0]
    features = cloud.prepare_features(sample['features'])
    output = cloud.forward(features)
    gamma = cloud.decode_assignment(output)
    plot_network(ap_pos, user_pos, gamma)
    
    # ---- Step 5: Summary ----
    print("\n[5/5] Done!")
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  QNN min rate:    {qnn_res['avg_min_rate']:.4f} bits/s/Hz")
    print(f"  Search min rate: {search_res['avg_min_rate']:.4f} bits/s/Hz")
    print(f"  Random min rate: {random_res['avg_min_rate']:.4f} bits/s/Hz")
    print(f"\n  Plots saved to: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()