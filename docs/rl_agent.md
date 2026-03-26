# FireGrid RL Tactical Agent — Architecture & Results

## What Is It?

The RL Tactical Agent is the second half of the FireGrid ML pipeline.

- **Stage 1 (XGBoost)** answers: *"How far will this fire spread?"*
- **Stage 2 (PPO Agent)** answers: *"Where should we deploy helicopters and ground crews to stop it?"*

---

## Architecture: Cellular Automata + PPO

### The Fire Simulator (`models/fire_env.py`)

The agent learns inside a 50×50 grid-based wildfire simulator built as a [`gymnasium`](https://gymnasium.farama.org/) environment — the standard interface for RL research.

```
Grid cell states:
  0 = Unburned fuel
  1 = Actively burning
  2 = Burned / scorched
  3 = Suppressed (firebreak / retardant)
```

**Key design:** The fire doesn't spread at a fixed rate. The spread probability per timestep is computed from the XGBoost output:
```python
spread_prob = min(0.85, xgboost_spread_1h_m / 250)
```
High ISI + low humidity = fast spread → harder game for the agent.

### Agent Actions (6 discrete)

| Action | Effect |
|---|---|
| Move N / S / E / W | Navigate to best tactical position |
| Deploy Helicopter (4) | Suppresses 3×3 cell area around agent with retardant |
| Deploy Ground Crew (5) | Places 1-cell firebreak at agent position |

### Reward Function

The agent is rewarded or penalized after every timestep:

| Event | Reward |
|---|---|
| Suppress a burning cell | **+3 per cell** |
| Proactive firebreak in fire's path | **+2** |
| Fire extinguished entirely | **+100 bonus** |
| Each new cell catching fire | **−0.5 per cell** |
| Wasting resources on burned land | **−3** |
| Wasted helicopter drop (no burning cells nearby) | **−2** |

### Policy: Proximal Policy Optimization (PPO)

PPO is the industry standard RL algorithm (same family as OpenAI's InstructGPT). It learns by playing thousands of fire simulations, updating its policy using the reward signal. Implemented via [`stable-baselines3`](https://stable-baselines3.readthedocs.io/).

**Observation space:** Flat 2,502-dimensional vector:
- 2,500 = 50×50 grid values (normalized 0–1)
- 2 = agent position (x, y)

**Policy network:** `MlpPolicy` (3-layer MLP, 64 hidden units per layer)

---

## Training Results

**Hyperparameters:**
- Total timesteps: 50,000
- Parallel environments: 4
- Learning rate: 3×10⁻⁴, γ = 0.995, ε_clip = 0.2
- ~2 minutes on CPU

**Reward curve (key milestones):**

| Timestep | Avg Episode Reward | Interpretation |
|---|---|---|
| 2,048 | **-84.8** | Agent wanders randomly, fire burns entire grid |
| 10,240 | **-14.8** | Agent starts locating fire and deploying |
| 26,624 | **+6.28** | Crosses zero — agent is net suppressing fire |
| 51,200 | **+22.6** | Agent consistently initiates suppression early |

The clean monotonic improvement from -84 to +22 indicates stable learning.

**Saved weights:** `backend/src/models/tactical_ppo_agent.zip`

---

## Inference & Greedy Safety Fallback

At inference time, the agent runs 60 deterministic steps in the simulator. Every time it takes a `Deploy Helicopter` or `Deploy Ground Crew` action, the grid cell is converted back to real-world lat/lon and added to the output.

**Why a fallback?** RL agents can sometimes fail to trigger a deployment action within the inference horizon, especially on novel initial states. To guarantee the frontend always has tactical lines to draw (critical for a live demo), the system includes a **Greedy Geometric Fallback**. If the PPO produces no deployments, it places 5 waypoints geometrically around the fire's 1-hour perimeter at cardinal directions.

---

## API

```
GET /api/v1/choke_points/{fire_id}
```

**Pipeline:**
1. Fetches fire record from DynamoDB (lat/lon, area)
2. Calls XGBoost → gets `spread_1h_m` to set fire danger level for the simulator
3. Runs PPO agent inference (or greedy fallback)
4. Returns:
```json
{
  "fire_id": "BC-2026-001",
  "spread_1h_m": 465,
  "spread_3h_m": 1541,
  "waypoints": [
    {
      "latitude": 49.912,
      "longitude": -119.501,
      "asset_type": "helicopter",
      "rationale": "PPO recommended helicopter deployment",
      "source": "ppo_agent"
    }
  ]
}
```

---

## Run It

```powershell
# Retrain the agent (optional — weights already saved)
uv run python -m src.models.train_rl_agent

# Quick 30-second training test
uv run python -m src.models.train_rl_agent --timesteps 10000

# Test via API (backend must be running)
Invoke-RestMethod http://localhost:8000/api/v1/choke_points/BC-2026-001
```
