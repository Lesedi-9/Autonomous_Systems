"""
plot_final.py
Component B - Final visualisation: consensus phase + assigned-target phase.
Reads epuck*_final_log.csv and assignment.csv from the swarm_controller folder.
"""

import csv
import os
import matplotlib.pyplot as plt

BASE = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\swarm_controller"
AGENTS = ["epuck1", "epuck2", "epuck3"]
COLORS = {"epuck1": "tab:red", "epuck2": "tab:blue", "epuck3": "tab:green"}

# ------------------------------------------------------------------
# Load the PSO assignment
# ------------------------------------------------------------------
assignment = {}
assign_path = os.path.join(BASE, "assignment.csv")
if os.path.exists(assign_path):
    with open(assign_path) as f:
        r = csv.reader(f)
        next(r, None)  # skip header safely
        for row in r:
            if len(row) >= 4:
                assignment[row[0]] = (float(row[2]), float(row[3]))
    print(f"Loaded assignment: {assignment}")
else:
    print(f"WARNING: {assign_path} not found")

# ------------------------------------------------------------------
# Load each agent's trajectory and split by phase
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))

for agent in AGENTS:
    t, x, y, phase = [], [], [], []
    path = os.path.join(BASE, f"{agent}_final_log.csv")

    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            print(f"Empty file (no header): {path}")
            continue
        for row in r:
            if len(row) < 4:
                continue
            try:
                t.append(float(row[0]))
                x.append(float(row[1]))
                y.append(float(row[2]))
                phase.append(int(row[3]))
            except ValueError:
                continue

    if not x:
        print(f"No data rows in: {path}")
        continue

    print(f"{agent}: {len(x)} points, phases = {set(phase)}")

    phase1_x = [x[i] for i in range(len(x)) if phase[i] == 1]
    phase1_y = [y[i] for i in range(len(y)) if phase[i] == 1]
    phase2_x = [x[i] for i in range(len(x)) if phase[i] == 2]
    phase2_y = [y[i] for i in range(len(y)) if phase[i] == 2]

    # Phase 1: dashed (consensus), Phase 2: solid (navigation)
    if phase1_x:
        ax.plot(phase1_x, phase1_y, '--', color=COLORS[agent],
                linewidth=1.5, alpha=0.6)
    if phase2_x:
        ax.plot(phase2_x, phase2_y, '-', color=COLORS[agent],
                linewidth=2.5, label=f'{agent}')

    # Start marker (circle) and end marker (X)
    ax.plot(x[0], y[0], 'o', color=COLORS[agent],
            markersize=14, markeredgecolor='black')
    ax.plot(x[-1], y[-1], 'X', color=COLORS[agent],
            markersize=16, markeredgecolor='black')

# ------------------------------------------------------------------
# Draw the three target positions
# ------------------------------------------------------------------
TARGETS = {"target1": (-2.0, 1.5),
           "target2": ( 2.0, 1.5),
           "target3": ( 0.0, -2.0)}
for tname, (tx, ty) in TARGETS.items():
    ax.plot(tx, ty, 's', color='green',
            markersize=20, markeredgecolor='black')
    ax.text(tx, ty + 0.25, tname, ha='center',
            fontsize=10, fontweight='bold')

ax.set_title('Component B: Consensus phase (dashed) → Assignment phase (solid)')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')

plt.tight_layout()
plt.show()