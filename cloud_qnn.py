# cloud_qnn.py
"""
Cloud QNN for Transmitter-User Assignment (Section IV-A).

Takes complete channel information from all APs.
Outputs which AP should serve which user.

Architecture (Eq. 10):
    U_cloud = U_connect(theta) · U_encode(H)

    Encoding:   RZ(channel_data) + H  on each qubit
    Connection: RY(theta) + CZ + CX   (trainable)
    Output:     Measure last layer → decode into assignment
"""

import pennylane as qml
from pennylane import numpy as np
import config


class CloudQNN:
    def __init__(self):
        # Circuit dimensions
        self.n_layers = config.N_LAYER_CLOUD     # 3
        self.n_neurons = config.N_AP             # 4 (one output per AP)
        self.n_qubits = self.n_layers * self.n_neurons  # 3 × 4 = 12
        
        # Quantum device (simulator)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Trainable parameters: one angle per qubit
        # Shape: (n_layers, n_neurons)
        self.params = np.random.uniform(
            0, 2 * np.pi,
            size=(self.n_layers, self.n_neurons),
            requires_grad=True
        )
        
        # Build the quantum circuit as a QNode
        self.circuit = qml.QNode(self._circuit, self.dev, interface="autograd")
    
    def _encode(self, features):
        """
        Encoding operation (Eq. 11).
        
        For each qubit: RZ(data) then H
        
        RZ encodes channel data as phase rotation.
        H creates superposition so the data can interact
        with trainable gates later.
        """
        for i in range(self.n_qubits):
            if i < len(features):
                qml.RZ(features[i], wires=i)
            qml.Hadamard(wires=i)
    
    def _connect(self, params):
        """
        Connection operation (Eq. 12).

        Global gate ordering as in the paper:
          U_connect = U_CX · U_CZ · U_RY

        Applied to the state in this order (leftmost = first applied):
          1. All RY(θ) across every qubit in every layer — TRAINABLE weights
          2. All CZ chains within each layer                — intra-layer entanglement
          3. All CX links from last qubit of layer l to
             first qubit of layer l+1                      — inter-layer connections
        """
        # Step 1: All RY rotations (all layers, all neurons) — applied first
        for l in range(self.n_layers):
            for n in range(self.n_neurons):
                qubit = l * self.n_neurons + n
                qml.RY(params[l, n], wires=qubit)

        # Step 2: CZ chains within every layer — applied second
        for l in range(self.n_layers):
            for n in range(self.n_neurons - 1):
                q1 = l * self.n_neurons + n
                q2 = l * self.n_neurons + n + 1
                qml.CZ(wires=[q1, q2])

        # Step 3: CX inter-layer connections — applied last
        for l in range(self.n_layers - 1):
            q_last = l * self.n_neurons + self.n_neurons - 1
            q_next_first = (l + 1) * self.n_neurons
            qml.CNOT(wires=[q_last, q_next_first])
    
    def _circuit(self, params, features):
        """
        The complete cloud QNN circuit.
        
        All qubits start as |0⟩
        → Encode channel data (RZ + H)
        → Process through trainable gates (RY + CZ + CX)
        → Measure the output layer
        """
        # Encode channel information into qubits
        self._encode(features)
        
        # Apply trainable connection circuit
        self._connect(params)
        
        # Measure the LAST LAYER only (these are our outputs)
        # Last layer qubits: indices [8, 9, 10, 11] for 3 layers × 4 neurons
        last_layer_start = (self.n_layers - 1) * self.n_neurons
        output_wires = list(range(last_layer_start,
                                   last_layer_start + self.n_neurons))
        
        return qml.probs(wires=output_wires)
    
    def forward(self, features):
        """Run the circuit with current parameters."""
        return self.circuit(self.params, features)
    
    def prepare_features(self, H_matrix):
        """
        Convert channel feature matrix into QNN input (Eq. 11).

        H_matrix already contains interleaved real and imaginary parts
        (from channel_to_features).  We normalise the raw values to
        [0, 2π] so they can be used directly as RZ rotation angles,
        preserving sign/phase information (negative values map to the
        lower half of [0, π], positive values to the upper half).
        """
        flat = H_matrix.flatten()
        min_val = np.min(flat)
        max_val = np.max(flat)
        value_range = max_val - min_val + 1e-10
        return 2 * np.pi * (flat - min_val) / value_range
    
    def decode_assignment(self, probs):
        """
        Convert QNN output probabilities into assignment matrix γ.
        
        For each AP m:
            1. Compute marginal probability of qubit m being |1⟩
            2. Map to user index: user = floor(prob × N_USER)
            3. Set γ[m, user] = 1
        
        Also ensures every user gets at least one AP.
        """
        gamma = np.zeros((config.N_AP, config.N_USER))
        n_output = min(self.n_neurons, config.N_AP)
        
        for m in range(config.N_AP):
            if m < n_output:
                # Marginal probability of qubit m being |1⟩
                # Sum over all states where bit m is 1
                prob_one = 0.0
                n_states = len(probs)
                for state_idx in range(n_states):
                    # Check if bit m is 1 in this state
                    if (state_idx >> (n_output - 1 - m)) & 1:
                        prob_one += float(probs[state_idx])
                
                # Map probability [0,1] to user index [0, N_USER-1]
                user = int(np.floor(prob_one * config.N_USER))
                user = min(user, config.N_USER - 1)
                user = max(user, 0)
            else:
                user = m % config.N_USER
            
            gamma[m, user] = 1
        
        # Ensure every user has at least one AP (Eq. 6c)
        gamma = self._enforce_coverage(gamma)
        
        return gamma
    
    def _enforce_coverage(self, gamma):
        """Make sure no user is left without an AP."""
        for k in range(config.N_USER):
            if np.sum(gamma[:, k]) == 0:
                # Find an AP serving a user that has multiple APs
                ap_counts = np.sum(gamma, axis=1)
                user_ap_counts = np.sum(gamma, axis=0)
                
                # Find user with most APs
                rich_user = np.argmax(user_ap_counts)
                
                # Take one AP from that user
                aps_of_rich = np.where(gamma[:, rich_user] == 1)[0]
                if len(aps_of_rich) > 1:
                    m_reassign = aps_of_rich[0]
                    gamma[m_reassign, rich_user] = 0
                    gamma[m_reassign, k] = 1
                else:
                    # Last resort: just reassign any AP
                    m_reassign = np.random.randint(0, config.N_AP)
                    gamma[m_reassign, :] = 0
                    gamma[m_reassign, k] = 1
        
        return gamma
    
    def compute_target(self, H, rho):
        """
        Compute ideal target performance Φ_assign (reference point).
        
        Based on eigenvalue decomposition of channel matrix.
        This is the "best possible" performance we aim toward.
        """
        total = 0.0
        for m in range(config.N_AP):
            # Build channel matrix for AP m: (N_TX × N_USER)
            H_m = np.zeros((config.N_TX, config.N_USER), dtype=complex)
            for k in range(config.N_USER):
                H_m[:, k] = H[(m, k)]
            
            # Eigenvalues represent the "capacity" of each channel mode
            eigenvalues = np.linalg.eigvalsh(H_m @ H_m.conj().T)
            eigenvalues = np.maximum(eigenvalues, 0)
            N_lam = max(len(eigenvalues), 1)
            
            for lam in eigenvalues:
                total -= np.log2(1 + lam / N_lam * rho)
        
        return float(total)


# ---- Quick test ----
if __name__ == "__main__":
    from channel import generate_positions, compute_distances, generate_sample
    from rates import compute_mr_precoding, compute_user_rates, build_precoding_from_assignment
    
    # Generate channel data
    ap_pos, user_pos = generate_positions(seed=42)
    dist = compute_distances(ap_pos, user_pos)
    H, H_matrix = generate_sample(dist)
    
    # Create cloud QNN
    cloud = CloudQNN()
    
    print("=" * 50)
    print("Cloud QNN Test")
    print("=" * 50)
    print(f"  Qubits: {cloud.n_qubits}")
    print(f"  Trainable parameters: {cloud.params.size}")
    print(f"  Parameter shape: {cloud.params.shape}")
    
    # Prepare features
    features = cloud.prepare_features(H_matrix)
    print(f"\n  Input features: {len(features)} values")
    
    # Run forward pass
    output = cloud.forward(features)
    print(f"  Output shape: {output.shape}")
    print(f"  Output (first 8 probs): {np.round(output[:8], 4)}")
    
    # Decode assignment
    gamma = cloud.decode_assignment(output)
    print(f"\n  Decoded assignment γ:")
    print(f"  {gamma.astype(int)}")
    
    # Check: which user does each AP serve?
    for m in range(config.N_AP):
        user = np.argmax(gamma[m])
        print(f"    AP{m} → User{user}")
    
    # Compute rates with this assignment
    V = build_precoding_from_assignment(H, gamma)
    rates = compute_user_rates(H, gamma, V)
    print(f"\n  User rates: {np.round(rates, 4)}")
    print(f"  Min rate: {np.min(rates):.4f}")
    
    # Show the circuit
    print(f"\n--- Circuit Diagram ---")
    try:
        print(qml.draw(cloud.circuit)(cloud.params, features))
    except:
        print("  (Circuit too large to display in text)")
    
    print("\n  The QNN works! But with random weights,")
    print("  the assignment is random too. Training will fix this.")