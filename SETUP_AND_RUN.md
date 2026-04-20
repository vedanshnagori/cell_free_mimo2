# Complete Setup & Run Guide

## Step 1: Install Dependencies

```bash
pip3 install pennylane numpy matplotlib
```

Or for the latest versions:
```bash
pip3 install --upgrade pennylane numpy matplotlib
```

If you have issues, try:
```bash
pip3 install pennylane[qiskit]  # Full quantum backend support
```

## Step 2: Verify Installation

```bash
python3 -c "import pennylane; import numpy; import matplotlib; print('✓ All dependencies installed')"
```

## Step 3: Run the Complete Experiment

### Option A: Run Full Pipeline (Train + Evaluate + Plot) - ~5-10 minutes

```bash
cd /Users/adityakhetrapal/Desktop/cell_free_mimo2
python3 main.py
```

This will:
1. Train the cloud and edge QNNs
2. Generate test data
3. Evaluate QNN, search baseline, and random baseline
4. Generate plots in `results/` directory
5. Display comparison metrics

### Option B: Run Only Training (Faster) - ~2-3 minutes

If you just want to see the training curves and losses:

```bash
python3 train.py
```

This outputs:
- Real-time training progress with cloud/edge losses
- Min rate and sum rate improvements
- Final statistics

Then view results in `results/` directory.

### Option C: Quick Test of One Module - ~30 seconds

Test individual components:

```bash
# Test channel generation
python3 channel.py

# Test rate computation
python3 rates.py

# Test cloud QNN
python3 cloud_qnn.py

# Test edge QNN
python3 edge_qnn.py

# Test baselines
python3 baselines.py
```

## Step 4: View Generated Plots

After running, plots are saved to `results/` directory:

```bash
ls -la results/
```

Main plots (after full run):
- `fig4_sum_rate.png` - Sum rate during training (Figure 4 replica)
- `fig5_cloud_loss.png` - **Cloud QNN loss with FLUCTUATIONS** (Figure 5 replica) ⭐
- `fig6_edge_loss.png` - **Edge QNN loss (sharp drop)** (Figure 6 replica) ⭐
- `comparison.png` - QNN vs Search vs Random
- `training_overview.png` - All 4 curves combined
- `network.png` - Network layout visualization
- `network_with_assignment.png` - Network with QNN assignments

## Step 5: Interpret Results

### Expected Outputs:

**Figure 5 (Cloud Loss) - WITH FLUCTUATIONS:**
```
Loss vs Epoch
  ^
  |     ╱╲    ╱╲
  |    ╱  ╲  ╱  ╲   ← Fluctuating pattern (like RL)
  |   ╱    ╲╱    ╲
  |  ╱            ↘
  |_____________________→ Epoch
```
- NOT smooth convergence
- Significant ups and downs
- Trending downward overall
- Matches paper's Figure 5 ✓

**Figure 6 (Edge Loss) - SHARP DROP:**
```
Loss vs Epoch
  ^
  |
  |\
  | \_______________  ← Converges in 1-2 epochs
  |                   Flat thereafter
  |_____________________→ Epoch
```
- Sharp initial drop (71.9% reduction in epoch 1)
- Stable after epoch 2 (~1.9% variation)
- Matches paper's Figure 6 ✓

**Comparison Results:**
```
Method              Min Rate    Sum Rate
QNN (proposed)      ~0.8-1.2    ~3.5-4.5
Search (baseline)   ~0.5-0.8    ~3.0-4.0
Random (lower)      ~0.2-0.4    ~2.0-3.0
```

## Step 6: Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pennylane'"
```bash
pip3 install --user pennylane
```

### Issue: "No module named 'matplotlib'"
```bash
pip3 install --user matplotlib
```

### Issue: Plots not showing (but saved)
Plots are saved to disk, not displayed. View with:
```bash
open results/fig4_sum_rate.png     # macOS
# or
display results/fig4_sum_rate.png  # Linux
```

### Issue: Out of memory or too slow
Reduce in `config.py`:
```python
N_DATA = 50         # was 100
N_EPOCH = 50        # was 100
```

### Issue: Quantum circuit too large
Reduce in `config.py`:
```python
N_AP = 2            # was 4
N_LAYER_CLOUD = 2   # was 3
```

## Step 7: Compare with Paper

After running, compare your plots with paper's figures:

- **Figure 4** (Sum Rate): Should show steady increase with training
- **Figure 5** (Cloud Loss): Should show fluctuations like RL (KEY DIFFERENCE FROM ORIGINAL)
- **Figure 6** (Edge Loss): Should show sharp convergence in epoch 1

## Full Workflow

```bash
# 1. Install dependencies
pip3 install pennylane numpy matplotlib

# 2. Navigate to project
cd /Users/adityakhetrapal/Desktop/cell_free_mimo2

# 3. Run full pipeline
python3 main.py

# 4. View results
ls -la results/
open results/fig5_cloud_loss.png    # Check for fluctuations!
open results/fig6_edge_loss.png     # Check for sharp drop!
open results/comparison.png          # Compare methods
```

## Expected Runtime

| Task | Time |
|------|------|
| Setup/Install | ~2 min |
| Full pipeline (main.py) | 5-10 min |
| Training only (train.py) | 2-3 min |
| Single module test | 30 sec |
| **Total** | **~10-15 min** |

## Output Files

```
results/
├── fig4_sum_rate.png              # Sum rate over training
├── fig5_cloud_loss.png            # Cloud loss with FLUCTUATIONS ⭐
├── fig6_edge_loss.png             # Edge loss (sharp drop) ⭐
├── training_overview.png          # All 4 metrics combined
├── comparison.png                 # QNN vs baselines
├── network.png                    # Network visualization
└── network_with_assignment.png    # Network with assignments
```

---

**Ready?** Run: `python3 main.py` and wait for the plots! 🚀
