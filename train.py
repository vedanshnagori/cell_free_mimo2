# train.py
"""
Training Engine for the Multi-Stage QNN Protocol (Algorithm 1).

Implements the paper's loss functions exactly:

  Cloud QNN (Algorithm 2, Eq. 13-14):
      L_assign = ||Q_assign - Φ_assign||²
      Q_assign = -min_k R_k(γ_soft | v^MR)   [soft, differentiable]
      Φ_assign = Σ_m Φ_precode(ĥ_m)           [fixed eigenvalue reference]

  Edge QNN m (Algorithm 3, Eq. 15-16):
      L_precode = ||Q_precode - Φ_precode_m||²
      Q_precode = -R_{m→k}(v_m | ĥ_m, γ)     [per-AP SINR rate]
      Φ_precode = Φ_precode(ĥ_m)              [fixed eigenvalue reference]

Training is sequential (Algorithm 1):
  Phase 1 → train Cloud QNN completely
  Phase 2 → train each Edge QNN independently
"""

import time
import pennylane as qml
from pennylane import numpy as np
import config
from channel import generate_positions, compute_distances, generate_sample
from rates import (compute_mr_precoding, compute_user_rates,
                   build_precoding_from_assignment,
                   compute_phi_precode, compute_phi_assign)
from cloud_qnn import CloudQNN
from edge_qnn import EdgeQNN


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(N_samples, distances, seed=None):
    """Generate multiple channel samples for training."""
    if seed is not None:
        np.random.seed(seed)
    dataset = []
    for _ in range(N_samples):
        H, features = generate_sample(distances)
        dataset.append({'H': H, 'features': features})
    return dataset


# ---------------------------------------------------------------------------
# Helpers shared by both training phases
# ---------------------------------------------------------------------------

def _compute_marginals(full_probs, n_qubits):
    """
    Marginal probability of each qubit being |1⟩ from the joint
    probability distribution.  Fully differentiable (just sums).
    """
    marginals = []
    n_states = len(full_probs)
    for m in range(n_qubits):
        prob_one = 0.0
        for state_idx in range(n_states):
            if (state_idx >> (n_qubits - 1 - m)) & 1:
                prob_one = prob_one + full_probs[state_idx]
        marginals.append(prob_one)
    return marginals


def _smooth_min(rates):
    """
    Differentiable minimum via the identity
        min(a, b) = (a + b - |a - b|) / 2
    applied iteratively.
    """
    result = rates[0]
    for r in rates[1:]:
        result = (result + r - np.abs(result - r)) * 0.5
    return result


# ---------------------------------------------------------------------------
# Phase 1: Cloud QNN training
# ---------------------------------------------------------------------------

def _build_ap_gains(H, V_MR):
    """
    Pre-compute all channel gains |h_{m,k}^H v^MR_{m,j}|² for every
    (AP m, observed user k, transmitted-toward user j) triple.

    These are constants inside the cloud loss function.

    Returns gains[(m, k, j)] = float.
    """
    gains = {}
    for m in range(config.N_AP):
        for k in range(config.N_USER):
            for j in range(config.N_USER):
                h_mk = H[(m, k)]
                v_mj = V_MR[(m, j)]
                gains[(m, k, j)] = float(
                    np.abs(np.dot(np.conj(h_mk), v_mj)) ** 2
                )
    return gains


def _soft_Q_assign(marginals, gains, beta=10.0):
    """
    Differentiable Q_assign = -min_k R_k(γ_soft | v^MR).

    γ_soft[m, k] is a smooth (differentiable) approximation of the
    hard assignment produced by the cloud QNN:
        γ_soft[m, k] ∝ exp(-β · (p_m - centre_k)²)
    where p_m = marginals[m] ∈ [0, 1] and
    centre_k = (k + 0.5) / N_USER is the mid-point of bin k.

    The expected SINR at user k is then:
        signal_k     = Σ_m γ_soft[m,k] · ρ · |h_{m,k}^H v^MR_{m,k}|²
        interfere_k  = Σ_m Σ_{j≠k} γ_soft[m,j] · μ·ρ · |h_{m,k}^H v^MR_{m,j}|²
        R_k          = log2(1 + signal_k / (interfere_k + 1))
    """
    N_U = config.N_USER
    rates = []

    for k in range(N_U):
        signal = 0.0
        interference = 0.0

        for m in range(config.N_AP):
            p_m = marginals[m]  # autograd-tracked

            # Soft one-hot: score for each user j
            scores = [
                np.exp(-beta * (p_m - (j + 0.5) / N_U) ** 2)
                for j in range(N_U)
            ]
            total_score = sum(scores) + 1e-10

            gamma_mk = scores[k] / total_score  # P(AP m → user k)

            # Signal: AP m "beaming" toward user k
            signal = signal + gamma_mk * config.RHO * gains[(m, k, k)]

            # Interference: AP m beaming toward other user j ≠ k
            for j in range(N_U):
                if j != k:
                    gamma_mj = scores[j] / total_score
                    interference = (interference
                                    + gamma_mj * config.MU_NK
                                    * config.RHO * gains[(m, k, j)])

        R_k = np.log2(1.0 + signal / (interference + 1.0))
        rates.append(R_k)

    return -_smooth_min(rates)   # Q_assign = -min_k R_k


def train_cloud(cloud, dataset, n_epochs, lr_init):
    """
    Algorithm 2: Train the Cloud QNN.

    Loss: L_assign = (Q_assign(γ_soft | v^MR) − Φ_assign(Ĥ))²

    Returns per-epoch average loss and the achieved min/sum rates
    (computed with the hard-decoded γ for monitoring).
    """
    history = {'loss': [], 'min_rate': [], 'sum_rate': []}
    print("\n  [Phase 1] Training Cloud QNN …")

    for epoch in range(n_epochs):
        lr = float(lr_init / np.sqrt(epoch + 1))
        epoch_loss = []
        epoch_min_rate = []
        epoch_sum_rate = []

        for sample in dataset:
            H = sample['H']
            H_matrix = sample['features']

            # Fixed reference Φ_assign (constant for this channel sample)
            phi_assign = float(compute_phi_assign(H))

            # Pre-compute channel gains once (constants in the loss)
            V_MR = compute_mr_precoding(H)
            gains = _build_ap_gains(H, V_MR)

            cloud_features = cloud.prepare_features(H_matrix)
            n_out = cloud.n_neurons  # = N_AP

            def cloud_cost(params):
                """
                L_assign = (Q_assign − Φ_assign)²   (Eq. 13-14).

                Q_assign is computed from the soft assignment derived
                from the marginal probabilities of the output qubits.
                """
                full_probs = cloud.circuit(params, cloud_features)
                marginals = _compute_marginals(full_probs, n_out)
                Q = _soft_Q_assign(marginals, gains)
                return (Q - phi_assign) ** 2

            # Record loss before the update
            epoch_loss.append(float(cloud_cost(cloud.params)))

            # Gradient-descent step (parameter-shift rule inside QNode)
            opt = qml.GradientDescentOptimizer(stepsize=lr)
            cloud.params = opt.step(cloud_cost, cloud.params)

            # Monitor: evaluate with hard-decoded γ and MR precoding
            cloud_out = cloud.forward(cloud_features)
            gamma = cloud.decode_assignment(cloud_out)
            V = build_precoding_from_assignment(H, gamma)
            rates = compute_user_rates(H, gamma, V)
            epoch_min_rate.append(float(np.min(rates)))
            epoch_sum_rate.append(float(np.sum(rates)))

        history['loss'].append(float(np.mean(epoch_loss)))
        history['min_rate'].append(float(np.mean(epoch_min_rate)))
        history['sum_rate'].append(float(np.mean(epoch_sum_rate)))

        if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: "
                  f"L_assign={history['loss'][-1]:8.4f}  "
                  f"MinRate={history['min_rate'][-1]:6.3f}  "
                  f"SumRate={history['sum_rate'][-1]:6.3f}")

    return history


# ---------------------------------------------------------------------------
# Phase 2: Edge QNN training
# ---------------------------------------------------------------------------

def _edge_interference(H, m, k, gamma, V_MR):
    """
    Fixed interference at user k from all APs n ≠ m,
    where each AP n transmits with MR precoding toward its
    assigned user (as given by gamma).

        interference = Σ_{n≠m} μ·ρ·|h_{n,k}^H v^MR_{n, γ_n}|²

    This is a scalar constant during the optimisation of AP m.
    """
    interference = 0.0
    for n in range(config.N_AP):
        if n == m:
            continue
        assigned_n = np.where(gamma[n] == 1)[0]
        if len(assigned_n) == 0:
            continue
        k_n = int(assigned_n[0])
        h_nk = H[(n, k)]
        v_n = V_MR[(n, k_n)]
        interference += (config.MU_NK * config.RHO
                         * float(np.abs(np.dot(np.conj(h_nk), v_n)) ** 2))
    return interference


def _gain_from_outputs(outputs, h_mk):
    """
    Compute differentiable |h_{m,k}^H v_m|² where v_m is the
    normalised precoding vector decoded from the edge QNN outputs.

    Mapping (N_USER = 3 qubits, N_TX = 2 antennas):
        v_m[0] = outputs[0] + j·outputs[1]
        v_m[1] = outputs[2] + j·0            (imaginary part set to 0)
    Normalised so that ‖v_m‖ = 1.

    The squared norm of the unnormalised vector is:
        norm² = outputs[0]² + outputs[1]² + outputs[2]²

    Then |h^H v_m|² = |h^H v_m_unnorm|² / norm²
    which avoids any non-differentiable division.

    Expanded with h_mk = [a+ib, c+id]:
        Re(h^H v_unnorm) = a·e0 + b·e1 + c·e2
        Im(h^H v_unnorm) = a·e1 − b·e0 − d·e2
    """
    # N_USER = 3 qubits always produces exactly 3 outputs; the guard below
    # handles any future configuration change (e.g. N_USER < 3).
    e0 = outputs[0]
    e1 = outputs[1]
    e2 = outputs[2] if len(outputs) > 2 else 0.0

    a = float(np.real(h_mk[0]))
    b = float(np.imag(h_mk[0]))
    c = float(np.real(h_mk[1]))
    d = float(np.imag(h_mk[1]))

    re_inner = a * e0 + b * e1 + c * e2
    im_inner = a * e1 - b * e0 - d * e2
    numerator = re_inner ** 2 + im_inner ** 2

    norm_sq = e0 ** 2 + e1 ** 2 + e2 ** 2 + 1e-10

    return numerator / norm_sq   # |h^H v_normalised|²


def train_edges(edges, cloud, dataset, n_epochs, lr_init):
    """
    Algorithm 3 (applied independently per AP): Train the Edge QNNs.

    For AP m assigned to user k by the cloud:
        Loss: L_precode = (Q_precode_m − Φ_precode_m)²
        Q_precode_m = -R_{m→k}(v_m | ĥ_m, γ)   (per-AP SINR rate)
        Φ_precode_m = Φ_precode(ĥ_m)             (eigenvalue reference)

    All APs n ≠ m are treated as interference with their MR precodings.
    """
    all_history = {m: {'loss': [], 'min_rate': [], 'sum_rate': []}
                   for m in range(config.N_AP)}
    print("\n  [Phase 2] Training Edge QNNs …")

    for m in range(config.N_AP):
        edge = edges[m]
        print(f"    AP {m}:")

        for epoch in range(n_epochs):
            lr = float(lr_init / np.sqrt(epoch + 1))
            epoch_loss = []
            epoch_min_rate = []
            epoch_sum_rate = []

            for sample in dataset:
                H = sample['H']
                V_MR = compute_mr_precoding(H)

                # Get the assignment from the (already trained) cloud
                H_matrix = sample['features']
                cloud_feat = cloud.prepare_features(H_matrix)
                cloud_out = cloud.forward(cloud_feat)
                gamma = cloud.decode_assignment(cloud_out)

                assigned = np.where(gamma[m] == 1)[0]
                if len(assigned) == 0:
                    # AP m has no assignment; skip
                    continue
                k_assigned = int(assigned[0])

                # Fixed reference Φ_precode for AP m
                phi_precode = float(compute_phi_precode(H, m))

                # Constant interference from other APs (fixed during AP-m update)
                fixed_interf = _edge_interference(H, m, k_assigned,
                                                  gamma, V_MR)

                edge_features = edge.prepare_features(H)
                h_mk = H[(m, k_assigned)]

                def edge_cost(params, feat=edge_features,
                              h=h_mk, interf=fixed_interf,
                              phi=phi_precode, eq=edge):
                    """
                    L_precode = (Q_precode − Φ_precode)²   (Eq. 15-16).

                    Q_precode = -log2(1 + ρ|h_{m,k}^H v_m|²/(interf+1))
                    where v_m is decoded from the circuit's PauliZ outputs.
                    """
                    outputs = eq.circuit(params, feat)
                    gain = _gain_from_outputs(outputs, h)
                    sinr = config.RHO * gain / (interf + 1.0)
                    R_mk = np.log2(1.0 + sinr)
                    Q = -R_mk
                    return (Q - phi) ** 2

                epoch_loss.append(float(edge_cost(edge.params)))

                opt = qml.GradientDescentOptimizer(stepsize=lr)
                edge.params = opt.step(edge_cost, edge.params)

                # Monitor: full-network rates using all edges current output
                V_eval = {}
                for n in range(config.N_AP):
                    assigned_n = np.where(gamma[n] == 1)[0]
                    if len(assigned_n) > 0:
                        ef = edges[n].prepare_features(H)
                        eo = edges[n].forward(ef)
                        V_eval[n] = edges[n].decode_precoding(eo)
                    else:
                        V_eval[n] = np.zeros(config.N_TX, dtype=complex)
                rates = compute_user_rates(H, gamma, V_eval)
                epoch_min_rate.append(float(np.min(rates)))
                epoch_sum_rate.append(float(np.sum(rates)))

            hist = all_history[m]
            hist['loss'].append(float(np.mean(epoch_loss)) if epoch_loss else 0.0)
            hist['min_rate'].append(float(np.mean(epoch_min_rate)) if epoch_min_rate else 0.0)
            hist['sum_rate'].append(float(np.mean(epoch_sum_rate)) if epoch_sum_rate else 0.0)

            if epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0:
                print(f"      Epoch {epoch+1:3d}/{n_epochs}: "
                      f"L_precode={hist['loss'][-1]:8.4f}  "
                      f"MinRate={hist['min_rate'][-1]:6.3f}  "
                      f"SumRate={hist['sum_rate'][-1]:6.3f}")

    return all_history


# ---------------------------------------------------------------------------
# Main training entry point (Algorithm 1)
# ---------------------------------------------------------------------------

def train():
    """
    Algorithm 1: Sequential multi-stage QNN training.

      1. Train Cloud QNN  → optimised assignment γ*
      2. Train Edge QNNs  → optimised precoding v*_m for each AP
    """
    print("=" * 60)
    print("SETTING UP")
    print("=" * 60)

    ap_pos, user_pos = generate_positions(seed=config.SEED)
    distances = compute_distances(ap_pos, user_pos)

    print(f"  Network: {config.N_AP} APs, {config.N_USER} users")
    print(f"  Distances:\n{np.round(distances, 3)}")

    dataset = generate_dataset(config.N_DATA, distances, seed=config.SEED)
    print(f"  Training samples: {config.N_DATA}")
    print(f"  Epochs per phase: {config.N_EPOCH}")

    cloud = CloudQNN()
    edges = [EdgeQNN(m) for m in range(config.N_AP)]

    print(f"\n  Cloud QNN : {cloud.n_qubits} qubits, {cloud.params.size} params")
    print(f"  Edge QNNs : {edges[0].n_qubits} qubits, "
          f"{edges[0].params.size} params each")

    total_start = time.time()

    # ── Phase 1: Cloud ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1: CLOUD QNN TRAINING")
    print("=" * 60)
    cloud_hist = train_cloud(cloud, dataset, config.N_EPOCH, config.LR)

    # ── Phase 2: Edges ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: EDGE QNN TRAINING")
    print("=" * 60)
    edge_hist = train_edges(edges, cloud, dataset, config.N_EPOCH, config.LR)

    total_time = time.time() - total_start

    # Merge histories into the format expected by plot.py
    # Use the average edge loss/rate across APs for the legacy keys
    n_ep = config.N_EPOCH
    avg_edge_loss = [
        float(np.mean([edge_hist[m]['loss'][e] for m in range(config.N_AP)]))
        for e in range(n_ep)
    ]
    avg_min_rate = [
        float(np.mean([edge_hist[m]['min_rate'][e] for m in range(config.N_AP)]))
        for e in range(n_ep)
    ]
    avg_sum_rate = [
        float(np.mean([edge_hist[m]['sum_rate'][e] for m in range(config.N_AP)]))
        for e in range(n_ep)
    ]

    history = {
        'cloud_loss': cloud_hist['loss'],
        'edge_loss':  avg_edge_loss,
        'min_rate':   avg_min_rate,
        'sum_rate':   avg_sum_rate,
    }

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final cloud loss:  {cloud_hist['loss'][-1]:.4f}")
    print(f"  Final edge loss:   {avg_edge_loss[-1]:.4f}")
    print(f"  Final min rate:    {avg_min_rate[-1]:.4f}")
    print(f"  Final sum rate:    {avg_sum_rate[-1]:.4f}")

    return cloud, edges, history, distances, ap_pos, user_pos


if __name__ == "__main__":
    cloud, edges, history, distances, ap_pos, user_pos = train()
