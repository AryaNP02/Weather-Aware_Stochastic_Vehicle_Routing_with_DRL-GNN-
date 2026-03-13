# Deep Reinforcement Learning for Stochastic Vehicle Routing under Weather-Driven Demand Uncertainty

An end-to-end trainable vehicle routing agent built in PyTorch for the academic project for team 32 . The agent uses a **Graph Attention Network (GAT)** to encode customer and vehicle states, and a **REINFORCE policy gradient** with a learned critic baseline to construct routing tours under stochastic, weather-correlated customer demand -- where the true demand at each stop is only revealed upon arrival.

---

## Problem Statement

The Vehicle Routing Problem (VRP) is NP-hard. In real-world logistics, the problem is harder still: customer demand is rarely known precisely at dispatch time. It fluctuates with weather, seasonality, and other unobservable factors. Classical solvers (OR-Tools, CPLEX, hand-crafted heuristics) assume demand is deterministic, making them brittle when actual conditions diverge from the plan.

The **Stochastic Vehicle Routing Problem (SVRP)** captures this reality. Each customer has an observable **base demand**, but the **actual demand** realized upon arrival depends on:

- A latent **weather context vector** (temperature, humidity, wind speed) generated per episode.
- **Spatially correlated sensitivity**: nodes in different geographic zones react differently to weather conditions (e.g., northern nodes are more affected by heat, southern nodes by rain).
- **Random noise**: unpredictable fluctuations in demand.

The real demand follows:

```
D_real = D_base + alpha^T . W . alpha + noise
```

where `alpha` encodes the node-specific weather sensitivity matrix and `W` is the global weather vector, computed via Einstein summation over the batch. If a vehicle arrives at a customer with insufficient remaining capacity, a **failure penalty** is applied and the customer remains partially unserved. The agent is penalized for both long routes and demand failures, and must learn to manage this risk without ever observing `D_real` in advance.

---

## What This Project Solves

Given a set of customer nodes on a 2D plane, a depot, a vehicle with fixed capacity, and a stochastic weather context:

- The trained agent constructs a tour that **minimizes total route cost** (Euclidean travel distance + capacity-failure penalties) under demand uncertainty.
- Actual demand is **revealed only upon arrival** at each customer node.
- The policy runs in **near real-time at inference** (~0.04s per instance on GPU), requiring no per-instance re-optimization.
- The agent **generalizes across random problem instances** -- it does not memorize solutions but learns a general routing strategy.
- Achieves **100% completion rate** on all evaluation instances (all customer demands fully met).

---

## Key Features

- **Stochastic demand environment** -- weather-correlated, spatially heterogeneous demand with configurable noise. Supports a deterministic mode for controlled ablation.
- **Graph Attention Network encoder** -- 4-layer, 8-head GAT with residual connections and LayerNorm for stable deep propagation. Encodes customer coordinates, observable demand, weather vector, and a completion mask into dense node embeddings.
- **Attention-based pointer decoder** -- query-key attention mechanism between vehicle embeddings and node embeddings, with dynamic masking (visited nodes, empty vehicles forced to depot, depot blocked when capacity remains and customers are unserved).
- **REINFORCE with learned critic baseline** -- 4-layer MLP value network estimates episode returns; advantage normalization and entropy regularization stabilize training.
- **Adaptive training** -- ReduceLROnPlateau scheduler halves learning rate after 50 epochs of stagnation; gradient clipping at norm 1.0 prevents gradient explosion.
- **Three inference strategies** -- greedy decoding (fastest), beam search (balanced), and random sampling with best-of-N selection (most thorough).
- **Shared validation dataset** -- 1,000 serialized instances per problem size ensure reproducible, comparable evaluation across runs.
- **Automated output** -- training metric plots (reward and loss curves) and route visualizations are saved after every training and evaluation run.
- **CLI-driven pipeline** -- all hyperparameters, model paths, and modes are configurable via command-line arguments with sensible defaults.

---

## Technology Stack

| Layer | Tool |
|---|---|
| Deep Learning Framework | PyTorch |
| Graph Neural Network | Graph Attention Network (multi-head, residual connections, LayerNorm) |
| RL Algorithm | REINFORCE (policy gradient) with learned critic baseline |
| Inference Strategies | Greedy decoding, Beam Search, Random Sampling (best-of-N) |
| LR Scheduling | ReduceLROnPlateau (factor 0.5, patience 50) |
| Data Serialization | Python pickle (shared validation datasets) |
| Visualization | Matplotlib |
| Progress Tracking | tqdm |
| Logging | Python logging (structured, file + console) |
| Hardware | CUDA-compatible GPU (CPU fallback supported) |

---

## Architecture

### System Flow

```
                      +---------------------------+
                      |     Problem Instance      |
                      | (node coords, base demand,|
                      |  weather vector, capacity) |
                      +-------------+-------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
     +------------v-----------+          +------------v-----------+
     |  Customer Feature Enc  |          |  Vehicle Feature Enc   |
     |  (Linear -> GAT x4)    |          |  (Linear -> ReLU)      |
     |  residual + LayerNorm  |          |  position + load       |
     |  per layer, 8 heads    |          +------------+-----------+
     +------------+-----------+                       |
                  |                                   |
                  +--------+    +---------------------+
                           |    |
                  +--------v----v--------+
                  |  Attention Pointer   |
                  |  Q = W_q(vehicle)    |
                  |  K = W_k(customer)   |
                  |  scores = Q . K^T    |
                  |  + dynamic masking   |
                  |  softmax -> probs    |
                  +----------+-----------+
                             |
                  +----------v-----------+
                  |   Action Selection   |
                  |   greedy / beam /    |
                  |   multinomial sample |
                  +----------+-----------+
                             |
                  +----------v-----------+
                  |   Environment Step   |
                  |   - move vehicle     |
                  |   - reveal D_real    |
                  |   - deliver / fail   |
                  |   - compute reward   |
                  +----------+-----------+
                             |
                  +----------v-----------+
                  |   REINFORCE Update   |
                  |   - discounted       |
                  |     returns (g=0.99) |
                  |   - critic baseline  |
                  |   - advantage norm   |
                  |   - entropy reg      |
                  |   - grad clip (1.0)  |
                  +----------------------+
```

### Component Details

**GAT Encoder** (`graph_attention.py`): Projects raw customer features (7D: demand, weather[3], coords[2], mask) into `embedding_dim` via a linear layer, then passes through `n_layers` Graph Attention layers. Each layer computes multi-head self-attention over all node pairs, concatenates head outputs, adds a residual connection from the input, applies LayerNorm, and activates with ReLU. This enables information propagation across the full customer graph while maintaining gradient flow in deep (4+ layer) networks.

**Critic Baseline** (`baseline_network.py`): A 4-layer MLP that takes the mean-pooled customer features and mean-pooled vehicle features as input, and outputs a scalar value estimate for the current state. Trained with MSE loss against discounted returns to reduce policy gradient variance.

**Training Loop** (`policy_gradient.py`): Each epoch generates a batch (default 512) of random problem instances, rolls out the policy to completion (or timeout at `num_nodes * 3` steps), collects per-step rewards, computes discounted returns (gamma=0.99), and applies a REINFORCE update with advantage normalization and entropy regularization. The learning rate is halved by ReduceLROnPlateau after 50 epochs without reward improvement. Gradients are clipped at norm 1.0.

**Inference** (`search_strategies.py`): A pre-trained checkpoint is loaded and evaluated on the shared validation dataset with no gradient computation. Three strategies are available:
- **Greedy**: Always picks the highest-probability node. Fastest.
- **Beam Search**: Maintains top-k partial solutions at each step. Re-evaluates final candidates by replaying routes to get true costs.
- **Random Sampling**: Draws N stochastic rollouts from the policy distribution, returns the lowest-cost solution. Most thorough.

---

## Project Structure

```
.
+-- run_pipeline.py                  # Entry point: training + evaluation
+-- data_ingestion/
|   +-- data_manager.py              # Dataset load/save utilities
|   +-- automate_data_tasks.sh       # Shell automation for data tasks
|   +-- validation_dataset_n10.pkl   # Serialized validation data (10 nodes)
+-- src/
|   +-- config_manager.py            # CLI argument parser (all hyperparams)
|   +-- environment/
|   |   +-- routing_env.py           # SVRP environment (demand model, transitions, rewards)
|   +-- agents/
|   |   +-- routing_policy.py        # Policy network (GAT encoder + attention pointer)
|   |   +-- graph_attention.py       # Multi-head GAT with residual + LayerNorm
|   |   +-- baseline_network.py      # 4-layer MLP critic (value estimator)
|   +-- trainers/
|   |   +-- policy_gradient.py       # REINFORCE trainer with critic, LR scheduling
|   +-- solvers/
|   |   +-- search_strategies.py     # Greedy, beam search, random sampling
|   +-- shared/
|       +-- logger.py                # Logging setup (file + console)
|       +-- plotter.py               # Route and metric visualization
|       +-- dataset_handler.py       # Validation dataset generation and loading
+-- trained_models_svrp_baseline/
|   +-- svrp_baseline_policy.pt      # Pre-trained policy weights (5000 epochs)
|   +-- svrp_baseline_critic.pt      # Pre-trained critic weights
|   +-- execution_log.txt            # Full training log
+-- results/
    +-- evaluation_greedy_search/     # Greedy evaluation output
    +-- evaluation_beam_search/       # Beam search evaluation output
    +-- evaluation_random_sampling/   # Random sampling evaluation output
```

---

## Installation

Requires Python 3.8+ and a CUDA-capable GPU for full-speed training. CPU fallback is supported.

```bash
pip install torch numpy matplotlib tqdm
```

---

## How to Run

### Training from Scratch

```bash
python run_pipeline.py \
  --num_nodes 10 \
  --epochs 5000 \
  --batch_size 512 \
  --lr 1e-4 \
  --baseline_lr 2e-4 \
  --embedding_dim 256 \
  --gat_layers 4 \
  --gat_heads 8 \
  --entropy_weight 0.05 \
  --save_dir ./trained_models_svrp_baseline \
  --cuda
```

### Evaluation with Pre-Trained Model

**Greedy (fastest):**
```bash
python run_pipeline.py \
  --test_only \
  --load_model ./trained_models_svrp_baseline/model_final \
  --num_nodes 10 --capacity 50 \
  --embedding_dim 256 --gat_layers 4 --gat_heads 8 \
  --inference greedy --test_size 50 \
  --save_dir results/evaluation_greedy_search \
  --cuda
```

**Beam Search (balanced):**
```bash
python run_pipeline.py \
  --test_only \
  --load_model ./trained_models_svrp_baseline/model_final \
  --num_nodes 10 --capacity 50 \
  --embedding_dim 256 --gat_layers 4 --gat_heads 8 \
  --inference beam --beam_width 5 --test_size 100 \
  --save_dir results/evaluation_beam_search \
  --cuda
```

**Random Sampling (most thorough):**
```bash
python run_pipeline.py \
  --test_only \
  --load_model ./trained_models_svrp_baseline/model_final \
  --num_nodes 10 --capacity 50 \
  --embedding_dim 256 --gat_layers 4 --gat_heads 8 \
  --inference random --num_samples 1000 --test_size 50 \
  --save_dir results/evaluation_random_sampling \
  --cuda
```

### Key CLI Flags

| Flag | Description | Default |
|---|---|---|
| `--num_nodes` | Number of customer nodes (excluding depot) | 20 |
| `--num_vehicles` | Number of vehicles | 1 |
| `--capacity` | Vehicle capacity | 50.0 |
| `--epochs` | Number of training episode batches | 100 |
| `--batch_size` | Episodes per gradient update | 32 |
| `--lr` | Policy network learning rate | 1e-4 |
| `--baseline_lr` | Critic network learning rate | 1e-3 |
| `--embedding_dim` | Latent dimension for node embeddings | 128 |
| `--gat_layers` | Depth of the GAT encoder | 2 |
| `--gat_heads` | Attention heads per GAT layer | 4 |
| `--entropy_weight` | Entropy regularization coefficient | 0.01 |
| `--inference` | Strategy: `greedy`, `beam`, or `random` | greedy |
| `--beam_width` | Beam width for beam search | 3 |
| `--num_samples` | Sample count for random sampling | 16 |
| `--test_size` | Number of evaluation instances | 10 |
| `--deterministic_env` | Disable weather noise (ablation mode) | off |
| `--cuda` | Enable GPU acceleration | off |
| `--save_dir` | Output directory for checkpoints and logs | checkpoints |
| `--load_model` | Path prefix for loading a trained model | -- |
| `--test_only` | Skip training, run evaluation only | off |

---

## Results

Baseline model trained for 5,000 epochs on 10-customer SVRP instances with stochastic demand active. Architecture: 4 GAT layers, 8 attention heads, embedding dimension 256, batch size 512.

| Metric | Value |
|---|---|
| Training reward convergence | ~476 (stabilized by epoch 300) |
| Greedy evaluation mean cost | 483.52 (over 50 held-out instances) |
| Avg inference time (greedy, GPU) | 0.044s per instance |
| Completion rate | 100% (all customer demands fully met) |

Training automatically generates:
- `<save_dir>/training_metrics.png` -- reward and loss curves over epochs.
- `<save_dir>/route_visualization_<strategy>.png` -- plotted routes for the best evaluation instance.

###  Assets

| Asset | Link |
|---|---|
| **Model Weights** (policy + critic model) | [Google Drive](https://drive.google.com/drive/folders/1QbUQGFvqmSRVqomN1gwI4ardYIAcTcOe?usp=sharing) |
| **Evaluation Results** (route plots, metrics) | [Google Drive](https://drive.google.com/drive/folders/1TyB29jgEnatflTGEsqXlGE2Vi4JHJDsU?usp=sharing) |

---

## References

1. Iklassov, Z., Sobirov, I., Solozabal, R., & Takac, M. (2023). **Reinforcement Learning for Solving Stochastic Vehicle Routing Problem.** *arXiv preprint arXiv:2311.07708.* [Paper](https://arxiv.org/abs/2311.07708)

2. Nazari, M., Oroojlooy, A., Snyder, L., & Takac, M. (2018). **Reinforcement Learning for Solving the Vehicle Routing Problem.** *Advances in Neural Information Processing Systems (NeurIPS).* [Paper](https://arxiv.org/abs/1802.04240)
