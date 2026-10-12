"""
plot_consensus.py
Visualises the average consensus convergence of 3 agents.
"""

import csv
import matplotlib.pyplot as plt

BASE = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\swarm_controller"
AGENTS = ["epuck1", "epuck2", "epuck3"]
COLORS = {"epuck1": "tab:red", "epuck2": "tab:blue", "epuck3": "tab:green"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: trajectories + vote convergence in the arena
ax = axes[0]

for agent in AGENTS:
    t, x, y, vx, vy = [], [], [], [], []
    with open(f"{BASE}/{agent}_consensus_log.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            t.append(float(row[0]))
            x.append(float(row[1])); y.append(float(row[2]))
            vx.append(float(row[3])); vy.append(float(row[4]))
    ax.plot(x, y, '-', color=COLORS[agent], linewidth=2, label=f'{agent} (robot)')
    ax.plot(vx, vy, '--', color=COLORS[agent], linewidth=1, alpha=0.7, label=f'{agent} (vote)')
    ax.plot(x[0], y[0], 'o', color=COLORS[agent], markersize=10, markeredgecolor='black')

# Compute the theoretical centroid
init_votes = {"epuck1": (2.0, 2.0), "epuck2": (-2.0, 2.0), "epuck3": (0.0, -2.0)}
cx = sum(v[0] for v in init_votes.values()) / 3
cy = sum(v[1] for v in init_votes.values()) / 3
ax.plot(cx, cy, 'k*', markersize=20, label='Theoretical centroid', markeredgecolor='white')

ax.set_title('Consensus convergence in the arena')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Panel 2: vote values over time
ax2 = axes[1]
for agent in AGENTS:
    t, vx, vy = [], [], []
    with open(f"{BASE}/{agent}_consensus_log.csv") as f:
        r = csv.reader(f); next(r)
        for row in r:
            t.append(float(row[0]))
            vx.append(float(row[3])); vy.append(float(row[4]))
    ax2.plot(t, vx, '-', color=COLORS[agent], linewidth=2, label=f'{agent} vote x')
    ax2.plot(t, vy, '--', color=COLORS[agent], linewidth=2, label=f'{agent} vote y')

ax2.axhline(cx, color='black', linestyle=':', alpha=0.5)
ax2.axhline(cy, color='black', linestyle=':', alpha=0.5)
ax2.set_title('Vote values over time (convergence)')
ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Vote value')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.show()