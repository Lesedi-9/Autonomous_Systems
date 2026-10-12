"""
occupancy_grid.py
Component A - Step 2 & 3: Occupancy grid + Dijkstra + A* pathfinding.

Builds a 50x50 occupancy grid of the 5x5 m Webots arena,
then runs Dijkstra and A* to find a shortest path from start to goal.
Compares the two algorithms and visualises the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
import math

# ==================================================================
# 1. Arena parameters
# ==================================================================
ARENA_SIZE = 5.0
CELL_SIZE = 0.1
GRID_SIZE = int(ARENA_SIZE / CELL_SIZE)   # 50

# Absolute path where the planner writes the path file
# The Webots controller will read this exact file
PATH_CSV = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\my_controller\path.csv"

# ==================================================================
# 2. Obstacles (x, y, radius) in metres
# ==================================================================
obstacles = [
    (-2.0,  0.0, 0.30),   # obstacle1 (left, dark)
    ( 0.0,  1.0, 0.30),   # obstacle2 (top, purple)
    ( 2.0,  0.0, 0.35),   # obstacle3 (right, grey)
    ( 0.0, -2.0, 0.30),   # obstacle4 (bottom, blue)
]

# ==================================================================
# 3. Start and goal (metres, Webots coordinates)
# ==================================================================
start_m = (-2.0, -2.0)
goal_m  = ( 2.0,  2.0)

# ==================================================================
# 4. Helper: metres -> grid cell
# ==================================================================
def m_to_cell(x, y):
    col = int((x + ARENA_SIZE / 2) / CELL_SIZE)
    row = int((y + ARENA_SIZE / 2) / CELL_SIZE)
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    return row, col

def cell_to_m(row, col):
    x = (col + 0.5) * CELL_SIZE - ARENA_SIZE / 2
    y = (row + 0.5) * CELL_SIZE - ARENA_SIZE / 2
    return x, y

# ==================================================================
# 5. Build the grid
# ==================================================================
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

for (ox, oy, r) in obstacles:
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cx, cy = cell_to_m(row, col)
            if (cx - ox) ** 2 + (cy - oy) ** 2 <= r ** 2:
                grid[row, col] = 1

# Outer walls
grid[0, :]  = 1
grid[-1, :] = 1
grid[:, 0]  = 1
grid[:, -1] = 1

# Start and goal in grid coordinates
start_cell = m_to_cell(*start_m)
goal_cell  = m_to_cell(*goal_m)

# Make sure start/goal aren't marked as obstacles
grid[start_cell] = 0
grid[goal_cell]  = 0

# ==================================================================
# 6. Neighbour moves (8-connected) and their costs
# ==================================================================
NEIGHBOURS = [
    (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
    (-1, -1, math.sqrt(2)), (-1,  1, math.sqrt(2)),
    ( 1, -1, math.sqrt(2)), ( 1,  1, math.sqrt(2)),
]

def neighbours(node):
    r, c = node
    for dr, dc, cost in NEIGHBOURS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            if grid[nr, nc] == 0:
                yield (nr, nc), cost

# ==================================================================
# 7. Dijkstra's Algorithm
# ==================================================================
def dijkstra(start, goal):
    """
    Returns (path, nodes_explored, elapsed_time).
    path is a list of (row, col) from start to goal, or None.
    """
    t0 = time.perf_counter()

    # Priority queue entries: (cost_so_far, node)
    pq = [(0.0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    visited = set()
    explored_count = 0

    while pq:
        current_cost, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        explored_count += 1

        if current == goal:
            break

        for nxt, step_cost in neighbours(current):
            new_cost = current_cost + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(pq, (new_cost, nxt))

    elapsed = time.perf_counter() - t0

    if goal not in came_from:
        return None, explored_count, elapsed

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path, explored_count, elapsed

# ==================================================================
# 8. A* Algorithm
# ==================================================================
def heuristic(a, b):
    """Euclidean distance in grid units."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def astar(start, goal):
    t0 = time.perf_counter()

    pq = [(heuristic(start, goal), 0.0, start)]  # (f, g, node)
    came_from = {start: None}
    cost_so_far = {start: 0.0}
    visited = set()
    explored_count = 0

    while pq:
        f, g, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        explored_count += 1

        if current == goal:
            break

        for nxt, step_cost in neighbours(current):
            new_cost = g + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                f_new = new_cost + heuristic(nxt, goal)
                heapq.heappush(pq, (f_new, new_cost, nxt))

    elapsed = time.perf_counter() - t0

    if goal not in came_from:
        return None, explored_count, elapsed

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path, explored_count, elapsed

# ==================================================================
# 9. Run both algorithms
# ==================================================================
# ==================================================================
# 9. Run both algorithms
# ==================================================================
path_d, explored_d, time_d = dijkstra(start_cell, goal_cell)
path_a, explored_a, time_a = astar(start_cell, goal_cell)

def path_length(path):
    if path is None:
        return None
    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i-1][0]
        dc = path[i][1] - path[i-1][1]
        total += math.hypot(dr, dc)
    return total

len_d = path_length(path_d)
len_a = path_length(path_a)

# Build the comparison table as a string
table_lines = [
    "=" * 55,
    f"{'Metric':<25}{'Dijkstra':<15}{'A*':<15}",
    "=" * 55,
    f"{'Path length (cells)':<25}{len_d:<15.2f}{len_a:<15.2f}",
    f"{'Nodes explored':<25}{explored_d:<15}{explored_a:<15}",
    f"{'Time (ms)':<25}{time_d*1000:<15.3f}{time_a*1000:<15.3f}",
    "=" * 55,
]
table_text = "\n".join(table_lines)

# Print to console (for the Run panel)
print(table_text)

# Also save to a text file
with open("comparison_results.txt", "w") as f:
    f.write(table_text)

# Also render the table as a figure so it's always visible
fig_tbl, ax_tbl = plt.subplots(figsize=(8, 3))
ax_tbl.axis('off')
ax_tbl.text(
    0.05, 0.95, table_text,
    fontsize=12, family='monospace',
    verticalalignment='top', transform=ax_tbl.transAxes,
)
ax_tbl.set_title("Dijkstra vs A* - Comparison", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# ==================================================================
# 10. Visualise both paths side by side
# ==================================================================
extent = [-ARENA_SIZE/2, ARENA_SIZE/2, -ARENA_SIZE/2, ARENA_SIZE/2]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, path, title, explored in [
    (axes[0], path_d, f'Dijkstra  (nodes explored: {explored_d})', explored_d),
    (axes[1], path_a, f'A*  (nodes explored: {explored_a})', explored_a),
]:
    ax.imshow(grid, cmap='Greys', origin='lower', extent=extent)

    if path:
        xs = [cell_to_m(r, c)[0] for (r, c) in path]
        ys = [cell_to_m(r, c)[1] for (r, c) in path]
        ax.plot(xs, ys, 'b-', linewidth=2, label='Path')
        ax.plot(xs, ys, 'b.', markersize=3)

    ax.plot(start_m[0], start_m[1], 'go', markersize=15,
            markeredgecolor='black', label='Start')
    ax.plot(goal_m[0], goal_m[1], 'ro', markersize=15,
            markeredgecolor='black', label='Goal')

    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_xticks(np.arange(-2.5, 3.0, 0.5))
    ax.set_yticks(np.arange(-2.5, 3.0, 0.5))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ==================================================================
# 11. Bar chart comparison
# ==================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels = ['Dijkstra', 'A*']
colors = ['steelblue', 'darkorange']

# Nodes explored
ax1.bar(labels, [explored_d, explored_a], color=colors)
ax1.set_title('Nodes Explored')
ax1.set_ylabel('Count')
for i, v in enumerate([explored_d, explored_a]):
    ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

# Time
ax2.bar(labels, [time_d*1000, time_a*1000], color=colors)
ax2.set_title('Computation Time')
ax2.set_ylabel('Time (ms)')
for i, v in enumerate([time_d*1000, time_a*1000]):
    ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()


# ==================================================================
# 12. Export the A* path to CSV for the Webots controller
# ==================================================================
import csv

if path_a is not None:
    with open(PATH_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])   # header
        for (r, c) in path_a:
            x, y = cell_to_m(r, c)
            writer.writerow([f"{x:.4f}", f"{y:.4f}"])
    print(f"\nA* path saved to path.csv ({len(path_a)} waypoints)")
else:
    print("\nNo path found — path.csv not written.")