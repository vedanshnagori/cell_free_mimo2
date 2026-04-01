# plot.py
"""
Plotting functions to reproduce Figures 4, 5, 6 from the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import config


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def plot_training_curves(history, save_dir="results"):
    """
    Plot training progress.
    Reproduces Figures 4, 5, 6 from the paper.
    """
    ensure_dir(save_dir)
    epochs = range(1, len(history['cloud_loss']) + 1)
    
    # --- Figure 4: Sum Rate over training ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['sum_rate'], 'b-o', markersize=3,
             linewidth=1.5, label=f'QNN (μ={config.LR})')
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Average Sum Rate (bits/s/Hz)', fontsize=12)
    plt.title('Fig 4: Achieved Sum Rate During Training', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_sum_rate.png'), dpi=150)
    plt.close()
    print(f"  Saved fig4_sum_rate.png")
    
    # --- Figure 5: Cloud Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['cloud_loss'], 'r-', linewidth=1.5,
             label=f'Cloud QNN (N_data={config.N_DATA})')
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Cloud Loss (L_assign)', fontsize=12)
    plt.title('Fig 5: Cloud QNN Training Loss', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_cloud_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved fig5_cloud_loss.png")
    
    # --- Figure 6: Edge Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history['edge_loss'], 'g-', linewidth=1.5,
             label=f'Edge QNN (N_data={config.N_DATA})')
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Edge Loss (L_precode)', fontsize=12)
    plt.title('Fig 6: Edge QNN Training Loss', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_edge_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved fig6_edge_loss.png")
    
    # --- Combined overview ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(epochs, history['cloud_loss'], 'r-', linewidth=1.5)
    axes[0, 0].set_ylabel('Cloud Loss')
    axes[0, 0].set_title('Cloud QNN Loss (↓ better)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['edge_loss'], 'g-', linewidth=1.5)
    axes[0, 1].set_ylabel('Edge Loss')
    axes[0, 1].set_title('Edge QNN Loss (↓ better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, history['min_rate'], 'b-', linewidth=1.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Min Rate (bits/s/Hz)')
    axes[1, 0].set_title('Minimum User Rate (↑ better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, history['sum_rate'], 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Sum Rate (bits/s/Hz)')
    axes[1, 1].set_title('Sum Rate (↑ better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Overview', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_overview.png'), dpi=150)
    plt.close()
    print(f"  Saved training_overview.png")


def plot_comparison(qnn_results, search_results, random_results,
                    save_dir="results"):
    """
    Bar chart comparing QNN vs baselines.
    """
    ensure_dir(save_dir)
    
    methods = ['QNN\n(proposed)', 'Search\n(baseline)', 'Random\n(lower bound)']
    min_rates = [qnn_results['avg_min_rate'], 
                 search_results['avg_min_rate'],
                 random_results['avg_min_rate']]
    sum_rates = [qnn_results['avg_sum_rate'],
                 search_results['avg_sum_rate'], 
                 random_results['avg_sum_rate']]
    colors = ['steelblue', 'coral', 'lightgreen']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(methods))
    
    axes[0].bar(x, min_rates, 0.5, color=colors, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=11)
    axes[0].set_ylabel('Min Rate (bits/s/Hz)', fontsize=12)
    axes[0].set_title('Minimum User Rate', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(min_rates):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    axes[1].bar(x, sum_rates, 0.5, color=colors, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, fontsize=11)
    axes[1].set_ylabel('Sum Rate (bits/s/Hz)', fontsize=12)
    axes[1].set_title('Sum Rate', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sum_rates):
        axes[1].text(i, v + 0.1, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.suptitle(f'Performance Comparison (N_AP={config.N_AP}, '
                 f'N_user={config.N_USER})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved comparison.png")


def plot_network(ap_pos, user_pos, gamma=None, save_dir="results"):
    """Visualize the network layout with optional assignment."""
    ensure_dir(save_dir)
    
    plt.figure(figsize=(7, 6))
    
    plt.scatter(ap_pos[:, 0], ap_pos[:, 1], c='red', marker='^', 
                s=200, label='APs', zorder=5, edgecolors='darkred')
    plt.scatter(user_pos[:, 0], user_pos[:, 1], c='blue', marker='o',
                s=150, label='Users', zorder=5, edgecolors='darkblue')
    
    for i in range(len(ap_pos)):
        plt.annotate(f'AP{i}', ap_pos[i] + [0.02, 0.02], fontsize=10,
                     fontweight='bold', color='red')
    for k in range(len(user_pos)):
        plt.annotate(f'U{k}', user_pos[k] + [0.02, -0.04], fontsize=10,
                     fontweight='bold', color='blue')
    
    if gamma is not None:
        colors = ['green', 'orange', 'purple', 'cyan']
        for m in range(len(ap_pos)):
            for k in range(len(user_pos)):
                if gamma[m, k] == 1:
                    plt.plot([ap_pos[m, 0], user_pos[k, 0]],
                             [ap_pos[m, 1], user_pos[k, 1]],
                             '--', color=colors[k % len(colors)],
                             linewidth=2, alpha=0.7)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cell-Free MIMO Network')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname = 'network_with_assignment.png' if gamma is not None else 'network.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()
    print(f"  Saved {fname}")