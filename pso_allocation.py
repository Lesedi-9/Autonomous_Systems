"""
pso_allocation.py
Component B - Step 3: PSO for assigning agents to targets.
Runs offline, writes assignment to assignment.csv
"""

import numpy as np
import csv
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
# Starting positions of the agents (must match world file)
AGENTS = {
    "epuck1": (-2.5, -2.5),
    "epuck2": ( 2.5, -2.5),
    "epuck3": ( 0.0,  2.5),
}

# Target positions (must match world file)
TARGETS = {
    "target1": (-2.0,  1.5),
    "target2": ( 2.0,  1.5),
    "target3": ( 0.0, -2.0),
}

agent_names  = list(AGENTS.keys())
target_names = list(TARGETS.keys())
n = len(agent_names)

# Cost matrix: cost[i][j] = distance agent_i to target_j
cost = np.zeros((n, n))
for i, a in enumerate(agent_names):
    for j, t in enumerate(target_names):
        ax, ay = AGENTS[a]
        tx, ty = TARGETS[t]
        cost[i][j] = np.hypot(ax - tx, ay - ty)

# ------------------------------------------------------------------
# PSO parameters
# ------------------------------------------------------------------
N_PARTICLES = 30
N_ITER      = 100
W           = 0.7    # inertia
C1          = 1.5    # cognitive (personal best)
C2          = 1.5    # social (global best)

rng = np.random.default_rng(42)

# Each particle is a permutation of target indices for each agent
# e.g. [2, 0, 1] means agent0→target2, agent1→target0, agent2→target1
def random_perm():
    return rng.permutation(n).tolist()

def evaluate(perm):
    """Total travel cost of this assignment."""
    return sum(cost[i][perm[i]] for i in range(n))

# Initialise particles
particles = [random_perm() for _ in range(N_PARTICLES)]
pbest     = [p.copy() for p in particles]
pbest_val = [evaluate(p) for p in particles]
gbest     = pbest[np.argmin(pbest_val)].copy()
gbest_val = min(pbest_val)

history = [gbest_val]

# ------------------------------------------------------------------
# PSO main loop
# ------------------------------------------------------------------
for it in range(N_ITER):
    for i in range(N_PARTICLES):
        # Update each position in the permutation by swapping toward
        # pbest and gbest with some probability.
        new_p = particles[i].copy()

        for k in range(n):
            r1 = rng.random()
            r2 = rng.random()

            if r1 < C1 / (C1 + C2):
                # Move toward pbest[k]
                target_val = pbest[i][k]
                if new_p[k] != target_val:
                    j = new_p.index(target_val)
                    new_p[k], new_p[j] = new_p[j], new_p[k]

            if r2 < C2 / (C1 + C2):
                target_val = gbest[k]
                if new_p[k] != target_val:
                    j = new_p.index(target_val)
                    new_p[k], new_p[j] = new_p[j], new_p[k]

        # Occasional random swap (exploration)
        if rng.random() < W * 0.3:
            a, b = rng.integers(0, n, size=2)
            new_p[a], new_p[b] = new_p[b], new_p[a]

        particles[i] = new_p
        v = evaluate(new_p)
        if v < pbest_val[i]:
            pbest[i]     = new_p.copy()
            pbest_val[i] = v
            if v < gbest_val:
                gbest     = new_p.copy()
                gbest_val = v

    history.append(gbest_val)

print(f"Best assignment: {gbest}")
print(f"Best total cost: {gbest_val:.3f} m")

# ------------------------------------------------------------------
# Save assignment for the controller
# ------------------------------------------------------------------
ASSIGN_CSV = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\swarm_controller\assignment.csv"
with open(ASSIGN_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["agent", "target", "target_x", "target_y"])
    for i, a in enumerate(agent_names):
        t = target_names[gbest[i]]
        tx, ty = TARGETS[t]
        w.writerow([a, t, f"{tx:.4f}", f"{ty:.4f}"])
print(f"Assignment saved to {ASSIGN_CSV}")

# ------------------------------------------------------------------
# Plot: convergence + assignment
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: PSO convergence
axes[0].plot(range(len(history)), history, 'b-', linewidth=2)
axes[0].set_title('PSO Convergence')
axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Best total cost (m)')
axes[0].grid(True, alpha=0.3)

# Panel 2: Assignment map
ax = axes[1]
for a, (ax_x, ax_y) in AGENTS.items():
    ax.plot(ax_x, ax_y, 'o', markersize=15, markeredgecolor='black', label=a)
for t, (tx, ty) in TARGETS.items():
    ax.plot(tx, ty, 's', color='green', markersize=15, markeredgecolor='black')
    ax.text(tx, ty + 0.15, t, ha='center', fontsize=10)

colors = ['tab:red', 'tab:blue', 'tab:green']
for i, a in enumerate(agent_names):
    ax_x, ax_y = AGENTS[a]
    tx, ty = TARGETS[target_names[gbest[i]]]
    ax.plot([ax_x, tx], [ax_y, ty], '-', color=colors[i], linewidth=2, alpha=0.7)

ax.set_title(f'Optimal Assignment (cost = {gbest_val:.2f} m)')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# Brute force check: confirm PSO found the true optimum
from itertools import permutations
best_brute = min(evaluate(list(p)) for p in permutations(range(n)))
print(f"\nBrute-force optimum: {best_brute:.3f} m")
print(f"PSO result:          {gbest_val:.3f} m")
print(f"PSO found optimum:   {abs(gbest_val - best_brute) < 1e-6}")