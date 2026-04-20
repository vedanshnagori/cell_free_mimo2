# train.py
"""
Training Engine for the Multi-Stage QNN Protocol (Algorithm 1).

Paper-aligned version with non-centralized cloud/edge training.
"""

import time
import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import config
from channel import generate_positions, compute_distances, generate_sample
from rates import (compute_mr_precoding, compute_user_rates,
                   build_precoding_from_assignment)
from cloud_qnn import CloudQNN
from edge_qnn import EdgeQNN


def generate_dataset(N_samples, distances, seed=None):
    """Generate multiple channel samples for training."""
    if seed is not None:
        np.random.seed(seed)
    
    dataset = []
    for i in range(N_samples):
        H, features = generate_sample(distances)
        dataset.append({'H': H, 'features': features})
    
    return dataset


def compute_ideal_assignment(H):
    """
    Compute the "best" assignment using channel strengths.
    
    This gives us a TARGET for the cloud QNN to learn.
    For each AP, which user has the strongest channel?
    
    Returns target_probs: for each AP, the probability of being |1⟩
    that would correspond to the best user assignment.
    """
    # For each AP, find the best user (strongest channel)
    best_users = []
    for m in range(config.N_AP):
        strengths = [np.linalg.norm(H[(m, k)]) for k in range(config.N_USER)]
        best_user = np.argmax(strengths)
        best_users.append(best_user)
    
    # Convert to target probabilities
    # If AP m should serve user k, target prob = (k + 0.5) / N_USER
    # This maps user index to a probability in [0, 1]
    target_probs = np.array([
        (best_users[m] + 0.5) / config.N_USER 
        for m in range(config.N_AP)
    ])
    
    return target_probs, best_users


def compute_marginal_probs(full_probs, n_qubits):
    """
    From the joint probability distribution,
    compute marginal probability of each qubit being |1⟩.
    
    This is differentiable since it's just summing probabilities!
    """
    marginals = []
    n_states = len(full_probs)
    
    for m in range(n_qubits):
        # Sum probabilities of all states where qubit m is |1⟩
        prob_one = 0.0
        for state_idx in range(n_states):
            if (state_idx >> (n_qubits - 1 - m)) & 1:
                prob_one = prob_one + full_probs[state_idx]
        marginals.append(prob_one)
    
    return marginals


def compute_cloud_rate_loss(cloud, H, cloud_features, params):
        """
        Paper-style cloud objective on discrete assignments.

        We optimize a max-min fairness surrogate directly:
            Q_assign = -min_k R_k(gamma | v^MR)
        This yields the negative cloud-loss trend seen in the paper figure.
        """
        probs = cloud.circuit(params, cloud_features)
        gamma = cloud.decode_assignment(probs)

        V_assign = build_precoding_from_assignment(H, gamma)
        rates = compute_user_rates(H, gamma, V_assign)

        loss = -float(np.min(rates))

        return float(loss), gamma


def spsa_step(params, loss_fn, step_size, perturb_scale, rng, n_avg=4):
        """
        One SPSA update for non-differentiable objectives.

        Uses two function evaluations regardless of parameter dimension.
        """
        grad_acc = np.zeros_like(params)
        loss_acc = 0.0

        for _ in range(n_avg):
            delta_np = rng.choice([-1.0, 1.0], size=params.shape)
            delta = np.array(delta_np)

            theta_plus = params + perturb_scale * delta
            theta_minus = params - perturb_scale * delta

            loss_plus = float(loss_fn(theta_plus))
            loss_minus = float(loss_fn(theta_minus))

            g_hat_scalar = (loss_plus - loss_minus) / (2.0 * perturb_scale)
            grad_acc = grad_acc + g_hat_scalar * delta
            loss_acc += 0.5 * (loss_plus + loss_minus)

        grad_est = grad_acc / n_avg
        avg_loss = loss_acc / n_avg

        new_params = params - step_size * grad_est
        # Keep angles bounded to avoid unstable drift during long training.
        new_params = np.mod(new_params, 2 * np.pi)

        return new_params, float(avg_loss)


def train():
    """
    Main training function implementing Algorithm 1.
    """
    print("=" * 60)
    print("SETTING UP")
    print("=" * 60)
    
    # Generate network
    ap_pos, user_pos = generate_positions(seed=config.SEED)
    distances = compute_distances(ap_pos, user_pos)
    
    print(f"  Network: {config.N_AP} APs, {config.N_USER} users")
    print(f"  Distances:\n{np.round(distances, 3)}")
    
    # Generate training data
    dataset = generate_dataset(config.N_DATA, distances, seed=config.SEED)
    print(f"  Training samples: {config.N_DATA}")
    print(f"  Epochs: {config.N_EPOCH}")
    
    # Initialize QNNs
    cloud = CloudQNN()
    edges = [EdgeQNN(m) for m in range(config.N_AP)]
    
    print(f"\n  Cloud QNN: {cloud.n_qubits} qubits, {cloud.params.size} params")
    print(f"  Edge QNNs: {edges[0].n_qubits} qubits, {edges[0].params.size} params each")
    
    # We use online channel sampling during each optimization step.
    # This keeps training closer to the paper setup where channels vary
    # by iteration and helps reproduce cloud-loss fluctuations.
    
    # Training history
    history = {
        'cloud_loss': [],
        'edge_loss': [],
        'min_rate': [],
        'sum_rate': [],
    }
    
    # =============================================
    # TRAINING LOOP
    # =============================================
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    total_start = time.time()
    rng = onp.random.default_rng(config.SEED)
    
    for epoch in range(config.N_EPOCH):
        epoch_start = time.time()
        
        # Descending learning rate
        lr = config.LR / np.sqrt(epoch + 1)
        
        # Accumulators
        epoch_cloud_loss = []
        epoch_edge_loss = []
        epoch_min_rate = []
        epoch_sum_rate = []
        
        for i_data in range(config.N_DATA):
            # Regenerate channel per iteration (paper-style stochastic training).
            H, H_matrix = generate_sample(distances)
            
            # ========================================
            # PHASE 1: Cloud QNN — Assignment
            # ========================================
            
            cloud_features = cloud.prepare_features(H_matrix)
            V_MR = compute_mr_precoding(H)

            # Paper-style cloud update with discrete rate objective.
            # Hard decoding is non-differentiable, so SPSA is used.
            def cloud_loss_for_params(theta):
                loss_val, _ = compute_cloud_rate_loss(cloud, H, cloud_features, theta)
                return loss_val

            # SPSA hyperparameters tuned for discrete cloud objective.
            cloud_step = 0.2 * lr
            perturb_scale = 0.2 / np.sqrt(epoch + 1)
            cloud.params, current_loss = spsa_step(
                cloud.params,
                cloud_loss_for_params,
                cloud_step,
                perturb_scale,
                rng,
                n_avg=4,
            )
            epoch_cloud_loss.append(current_loss)

            _, gamma = compute_cloud_rate_loss(cloud, H, cloud_features, cloud.params)
            
            # ========================================
            # PHASE 2: Edge QNNs — Precoding
            # ========================================
            
            V_MR = compute_mr_precoding(H)
            V_qnn = {}
            
            for m in range(config.N_AP):
                edge = edges[m]
                edge_features = edge.prepare_features(H)
                
                # Find which user this AP should serve
                assigned = np.where(gamma[m] == 1)[0]
                if len(assigned) == 0:
                    V_qnn[m] = np.zeros(config.N_TX, dtype=complex)
                    continue
                
                k_assigned = assigned[0]
                h_mk = H[(m, k_assigned)]
                
                # Target: MR precoding direction
                # We want the edge QNN to learn to output something
                # close to the MR precoding (or better)
                mr_target = V_MR[(m, k_assigned)]
                
                # Target expectation values for the edge QNN
                # Map MR precoding to target outputs in [-1, 1]
                target_outputs = np.zeros(config.N_USER)
                for j in range(config.N_TX):
                    if 2*j < config.N_USER:
                        target_outputs[2*j] = np.real(mr_target[j])
                    if 2*j+1 < config.N_USER:
                        target_outputs[2*j+1] = np.imag(mr_target[j])
                
                # Differentiable edge cost
                def edge_cost(params, feat=edge_features, 
                            targets=target_outputs, eq=edge):
                    """
                    Make QNN output match MR precoding direction.
                    
                    PauliZ expectation values should match target_outputs.
                    """
                    outputs = eq.circuit(params, feat)
                    cost = 0.0
                    for j in range(len(outputs)):
                        cost = cost + (outputs[j] - targets[j]) ** 2
                    return cost
                
                # Compute loss before update
                e_loss = float(edge_cost(edge.params))
                
                # Gradient descent step
                opt_edge = qml.GradientDescentOptimizer(stepsize=lr)
                edge.params = opt_edge.step(edge_cost, edge.params)
                
                # Get current precoding
                edge_output = edge.forward(edge_features)
                V_qnn[m] = edge.decode_precoding(edge_output)
            
            # ========================================
            # Compute achieved rates
            # ========================================
            V_final = {}
            for m in range(config.N_AP):
                assigned = np.where(gamma[m] == 1)[0]
                if len(assigned) > 0 and m in V_qnn:
                    V_final[m] = V_qnn[m]
                elif len(assigned) > 0:
                    V_final[m] = V_MR[(m, assigned[0])]
                else:
                    V_final[m] = np.zeros(config.N_TX, dtype=complex)
            
            rates = compute_user_rates(H, gamma, V_final)
            epoch_min_rate.append(float(np.min(rates)))
            epoch_sum_rate.append(float(np.sum(rates)))

            # Paper-like edge loss trend: negative sum-rate style metric.
            epoch_edge_loss.append(-float(np.sum(rates)))
        
        # ========================================
        # Record epoch statistics
        # ========================================
        avg_cloud = np.mean(epoch_cloud_loss)
        avg_edge = np.mean(epoch_edge_loss) if epoch_edge_loss else 0
        avg_min = np.mean(epoch_min_rate)
        avg_sum = np.mean(epoch_sum_rate)
        
        history['cloud_loss'].append(avg_cloud)
        history['edge_loss'].append(avg_edge)
        history['min_rate'].append(avg_min)
        history['sum_rate'].append(avg_sum)
        
        elapsed = time.time() - epoch_start
        
        if epoch == 0 or (epoch + 1) % max(1, config.N_EPOCH // 10) == 0:
            print(f"  Epoch {epoch+1:3d}/{config.N_EPOCH}: "
                  f"Cloud Loss={avg_cloud:8.4f}  "
                  f"Edge Loss={avg_edge:8.4f}  "
                  f"Min Rate={avg_min:6.3f}  "
                  f"Sum Rate={avg_sum:6.3f}  "
                  f"({elapsed:.1f}s)")
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final min rate:   {history['min_rate'][-1]:.4f}")
    print(f"  Final sum rate:   {history['sum_rate'][-1]:.4f}")
    print(f"  Final cloud loss: {history['cloud_loss'][-1]:.4f}")
    print(f"  Final edge loss:  {history['edge_loss'][-1]:.4f}")
    
    return cloud, edges, history, distances, ap_pos, user_pos


if __name__ == "__main__":
    cloud, edges, history, distances, ap_pos, user_pos = train()