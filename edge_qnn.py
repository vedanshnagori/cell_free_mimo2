# edge_qnn.py
"""
Edge QNN for Transmit Precoding Optimization (Section IV-B).

Each AP has its own Edge QNN.
Takes local channel info → outputs precoding vector.

Architecture (Eq. 19):
    U^[m] = U_connect(theta^[m]) · U_encode(h_m)

Much simpler than cloud QNN:
    - Only N_USER qubits (3 instead of 12)
    - Only local channel data (not full network)
    - Outputs precoding vector instead of assignment
"""

import pennylane as qml
from pennylane import numpy as np
import config


class EdgeQNN:
    def __init__(self, ap_index):
        """
        Create an Edge QNN for a specific AP.
        
        Args:
            ap_index: which AP this QNN belongs to (0, 1, 2, or 3)
        """
        self.ap_index = ap_index
        self.n_qubits = config.N_USER       # 3 qubits (one per user)
        self.n_layers = config.N_LAYER_EDGE  # 3 layers
        
        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Trainable parameters: (n_layers, n_qubits)
        self.params = np.random.uniform(
            0, 2 * np.pi,
            size=(self.n_layers, self.n_qubits),
            requires_grad=True
        )
        
        # Build QNode
        self.circuit = qml.QNode(self._circuit, self.dev, interface="autograd")
    
    def _encode(self, features):
        """
        Encode local channel info into qubits.
        
        Each qubit gets the channel strength to one user:
            qubit 0 ← channel to user 0
            qubit 1 ← channel to user 1
            qubit 2 ← channel to user 2
        """
        for k in range(self.n_qubits):
            if k < len(features):
                qml.RZ(features[k], wires=k)
            qml.Hadamard(wires=k)
    
    def _connect(self, params):
        """
        Trainable circuit (Eq. 20).
        
        Each layer:
            1. RY(θ) on each qubit — trainable weights
            2. CZ chain — entangle adjacent qubits
            3. CZ ring — connect last to first qubit
        
        Ring topology gives better entanglement than just a chain.
        """
        for l in range(self.n_layers):
            # Trainable rotations
            for k in range(self.n_qubits):
                qml.RY(params[l, k], wires=k)
            
            # CZ chain: 0-1, 1-2
            for k in range(self.n_qubits - 1):
                qml.CZ(wires=[k, k + 1])
            
            # CZ ring: close the loop (2-0)
            if self.n_qubits > 2:
                qml.CZ(wires=[self.n_qubits - 1, 0])
    
    def _circuit(self, params, features):
        """
        Full edge QNN circuit.
        
        Output: expectation value of PauliZ on each qubit.
        
        PauliZ expectation gives a value in [-1, 1]:
            +1 means qubit is definitely |0⟩
            -1 means qubit is definitely |1⟩
             0 means 50/50
        
        We use these values to build the precoding vector.
        """
        self._encode(features)
        self._connect(params)
        
        # Return expectation values (not probabilities!)
        # This gives us continuous values in [-1, 1]
        return [qml.expval(qml.PauliZ(k)) for k in range(self.n_qubits)]
    
    def forward(self, features):
        """Run the circuit."""
        return self.circuit(self.params, features)
    
    def prepare_features(self, H):
        """
        Extract local channel features for this AP.
        
        Takes the average channel magnitude to each user,
        scaled to [0, 2π] for RZ encoding.
        """
        features = np.array([
            np.abs(H[(self.ap_index, k)]).mean() * 2 * np.pi
            for k in range(config.N_USER)
        ])
        return features
    
    def decode_precoding(self, output):
        """
        Convert QNN output into complex precoding vector.
        
        output has N_USER values in [-1, 1].
        We pair them up as (real, imaginary) parts:
            v[0] = output[0] + j * output[1]
            v[1] = output[2] + j * 0  (if odd number)
        
        Then normalize so ||v||² = 1 (power constraint, Eq. 15b).
        """
        v = np.zeros(config.N_TX, dtype=complex)
        
        for j in range(config.N_TX):
            real_idx = 2 * j
            imag_idx = 2 * j + 1
            
            real_part = float(output[real_idx]) if real_idx < len(output) else 0.0
            imag_part = float(output[imag_idx]) if imag_idx < len(output) else 0.0
            
            v[j] = real_part + 1j * imag_part
        
        # Normalize: ||v||² ≤ 1
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            v = v / norm
        else:
            # Fallback: equal power across antennas
            v = np.ones(config.N_TX, dtype=complex) / np.sqrt(config.N_TX)
        
        return v


# ---- Quick test ----
if __name__ == "__main__":
    from channel import generate_positions, compute_distances, generate_sample
    from rates import compute_mr_precoding
    
    # Generate channel data
    ap_pos, user_pos = generate_positions(seed=42)
    dist = compute_distances(ap_pos, user_pos)
    H, H_matrix = generate_sample(dist)
    
    print("=" * 50)
    print("Edge QNN Test")
    print("=" * 50)
    
    # Create edge QNNs for all APs
    edge_qnns = [EdgeQNN(m) for m in range(config.N_AP)]
    
    # Also compute MR precoding for comparison
    V_MR = compute_mr_precoding(H)
    
    for m in range(config.N_AP):
        edge = edge_qnns[m]
        
        print(f"\n--- AP {m} Edge QNN ---")
        print(f"  Qubits: {edge.n_qubits}")
        print(f"  Parameters: {edge.params.size}")
        
        # Prepare features
        features = edge.prepare_features(H)
        print(f"  Input features: {np.round(features, 3)}")
        
        # Forward pass
        output = edge.forward(features)
        print(f"  Raw output: {[round(float(o), 4) for o in output]}")
        
        # Decode precoding
        v_qnn = edge.decode_precoding(output)
        v_mr = V_MR[(m, 0)]  # MR precoding toward user 0
        
        print(f"  QNN precoding:  {np.round(v_qnn, 3)}")
        print(f"  MR precoding:   {np.round(v_mr, 3)}")
        print(f"  ||v_qnn||² = {np.sum(np.abs(v_qnn)**2):.4f}")
    
    # Show circuit
    print(f"\n--- Edge QNN Circuit ---")
    edge0 = edge_qnns[0]
    features0 = edge0.prepare_features(H)
    print(qml.draw(edge0.circuit)(edge0.params, features0))
    
    print("\n  Edge QNNs are much simpler than Cloud QNN!")
    print(f"  Cloud: {12} qubits, {12} params")
    print(f"  Edge:  {config.N_USER} qubits, {config.N_USER * config.N_LAYER_EDGE} params each")