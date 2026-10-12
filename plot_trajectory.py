"""
plot_trajectory.py
Component A - Step 4: Visualise planned vs actual trajectory.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

PATH_CSV = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\my_controller\path.csv"
TRAJ_CSV = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\my_controller\actual_trajectory.csv"

# Load planned path
path_x, path_y = [], []
with open(PATH_CSV) as f:
    r = csv.reader(f); next(r)
    for row in r:
        path_x.append(float(row[0]))
        path_y.append(float(row[1]))

# Load actual trajectory
act_t, act_x, act_y = [], [], []
with open(TRAJ_CSV) as f:
    r = csv.reader(f); next(r)
    for row in r:
        act_t.append(float(row[0]))
        act_x.append(float(row[1]))
        act_y.append(float(row[2]))

# Plot
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(path_x, path_y, 'b--', linewidth=2, label='Planned (A*)')
ax.plot(act_x, act_y, 'r-', linewidth=2, label='Actual (PID)')
ax.plot(path_x[0], path_y[0], 'go', markersize=14, markeredgecolor='black', label='Start')
ax.plot(path_x[-1], path_y[-1], 'ko', markersize=14, markeredgecolor='black', label='Goal')

ax.set_title('Planned vs Actual Trajectory')
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# Tracking error metric
# Compare actual (x,y) to nearest planned waypoint
planned = np.array(list(zip(path_x, path_y)))
actual  = np.array(list(zip(act_x,  act_y)))
min_dist = [np.min(np.hypot(planned[:,0]-ax_, planned[:,1]-ay_)) for ax_, ay_ in actual]

print(f"Total time to goal: {act_t[-1]:.2f} s")
print(f"Path length (planned): {sum(np.hypot(np.diff(path_x), np.diff(path_y))):.3f} m")
print(f"Path length (actual):  {sum(np.hypot(np.diff(act_x), np.diff(act_y))):.3f} m")
print(f"Mean tracking error:   {np.mean(min_dist):.4f} m")
print(f"Max tracking error:    {np.max(min_dist):.4f} m")