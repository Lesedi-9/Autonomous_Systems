"""
swarm_controller.py
Component B - Step 3: Consensus + PSO-based task allocation.

Phase 1: agents reach consensus on a shared meeting point (average consensus).
Phase 2: agents navigate to their assigned target (from PSO).
"""

from controller import Robot
import math
import csv
import os
import time

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0
name = robot.getName()

START_POSES = {
    "epuck1": (-2.5, -2.5, 0.0),
    "epuck2": ( 2.5, -2.5, 0.0),
    "epuck3": ( 0.0,  2.5, 0.0),
}
INITIAL_VOTES = {
    "epuck1": ( 2.0,  2.0),
    "epuck2": (-2.0,  2.0),
    "epuck3": ( 0.0, -2.0),
}

x, y, theta = START_POSES[name]
vote_x, vote_y = INITIAL_VOTES[name]

# ------------------------------------------------------------------
# Load PSO assignment
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ASSIGN_CSV = os.path.join(HERE, "assignment.csv")

MY_TARGET = None
try:
    with open(ASSIGN_CSV) as f:
        r = csv.reader(f)
        next(r)  # skip header
        for row in r:
            if row[0] == name:
                MY_TARGET = (float(row[2]), float(row[3]))
                break
    if MY_TARGET:
        print(f"[{name}] Assigned target: ({MY_TARGET[0]:.2f},{MY_TARGET[1]:.2f})")
    else:
        print(f"[{name}] WARNING: no target found in assignment.csv")
except FileNotFoundError:
    print(f"[{name}] WARNING: assignment.csv not found - run pso_allocation.py first")

if MY_TARGET is None:
    MY_TARGET = (0.0, 0.0)

# ------------------------------------------------------------------
# Comms
# ------------------------------------------------------------------
COMM_DIR = os.path.join(HERE, "comms")
os.makedirs(COMM_DIR, exist_ok=True)
AGENTS = ["epuck1", "epuck2", "epuck3"]
OTHER_AGENTS = [a for a in AGENTS if a != name]
MY_FILE = os.path.join(COMM_DIR, f"{name}_vote.csv")
MY_TMP  = MY_FILE + ".tmp"

def write_vote(vx, vy):
    payload = f"{vx:.6f},{vy:.6f}\n"
    for _ in range(10):
        try:
            with open(MY_TMP, "w") as f:
                f.write(payload)
            os.replace(MY_TMP, MY_FILE)
            return
        except (PermissionError, OSError):
            time.sleep(0.001)

def read_vote(agent):
    path = os.path.join(COMM_DIR, f"{agent}_vote.csv")
    for _ in range(10):
        try:
            with open(path) as f:
                line = f.readline().strip()
            a, b = line.split(",")
            return float(a), float(b)
        except (FileNotFoundError, ValueError):
            return None
        except (PermissionError, OSError):
            time.sleep(0.001)
    return None

# ------------------------------------------------------------------
# Motors
# ------------------------------------------------------------------
left_motor  = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ------------------------------------------------------------------
# PID + kinematics
# ------------------------------------------------------------------
Kp, Ki, Kd = 2.5, 0.4, 0.6
integral = 0.0
prev_error = 0.0

MAX_SPEED    = 6.28
WHEEL_RADIUS = 0.0205
WHEELBASE    = 0.052
LINEAR_SPEED = 0.12
EPSILON      = 0.03

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOG_CSV = os.path.join(HERE, f"{name}_final_log.csv")
log_file = open(LOG_CSV, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["t", "x", "y", "phase", "goal_x", "goal_y"])
log_file.flush()

write_vote(vote_x, vote_y)
print(f"[{name}] Start pos=({x:.2f},{y:.2f})  vote=({vote_x:.2f},{vote_y:.2f})")

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
PHASE_CONSENSUS = 1
PHASE_NAVIGATE  = 2
phase = PHASE_CONSENSUS

last_log_t = -1.0
CONSENSUS_TIME = 8.0    # seconds spent on consensus before navigating
CONSENSUS_TOL  = 0.20   # convergence tolerance

while robot.step(timestep) != -1:
    t = robot.getTime()

    if phase == PHASE_CONSENSUS:
        neighbors = []
        for other in OTHER_AGENTS:
            v = read_vote(other)
            if v is not None:
                neighbors.append(v)

        if neighbors:
            dx_sum = sum(v[0] - vote_x for v in neighbors)
            dy_sum = sum(v[1] - vote_y for v in neighbors)
            vote_x += EPSILON * dx_sum
            vote_y += EPSILON * dy_sum

        write_vote(vote_x, vote_y)
        goal_x, goal_y = vote_x, vote_y

        # Move to phase 2 once consensus is reached
        if t > CONSENSUS_TIME and len(neighbors) == len(OTHER_AGENTS):
            spread = max(
                math.hypot(v[0] - vote_x, v[1] - vote_y) for v in neighbors
            )
            if spread < CONSENSUS_TOL:
                phase = PHASE_NAVIGATE
                goal_x, goal_y = MY_TARGET
                print(f"[{name}] Consensus reached, navigating to target "
                      f"({MY_TARGET[0]:.2f},{MY_TARGET[1]:.2f})")
    else:
        goal_x, goal_y = MY_TARGET

    # Navigate to (goal_x, goal_y)
    err_x = goal_x - x
    err_y = goal_y - y
    dist  = math.hypot(err_x, err_y)

    if dist > 0.10:
        desired_theta = math.atan2(err_y, err_x)
        error = desired_theta - theta
        while error >  math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi

        integral += error * dt
        derivative = (error - prev_error) / dt
        omega = Kp * error + Ki * integral + Kd * derivative
        prev_error = error

        speed_factor = max(0.2, 1.0 - abs(error) / math.pi)
        v_forward = LINEAR_SPEED * speed_factor

        v_l = v_forward - omega * WHEELBASE / 2.0
        v_r = v_forward + omega * WHEELBASE / 2.0

        w_l = max(-MAX_SPEED, min(MAX_SPEED, v_l / WHEEL_RADIUS))
        w_r = max(-MAX_SPEED, min(MAX_SPEED, v_r / WHEEL_RADIUS))
    else:
        w_l = w_r = 0.0

    left_motor.setVelocity(w_l)
    right_motor.setVelocity(w_r)

    # Odometry
    v_l_ms = w_l * WHEEL_RADIUS
    v_r_ms = w_r * WHEEL_RADIUS
    v = (v_l_ms + v_r_ms) / 2.0
    omega_actual = (v_r_ms - v_l_ms) / WHEELBASE

    x     += v * math.cos(theta) * dt
    y     += v * math.sin(theta) * dt
    theta += omega_actual * dt
    while theta >  math.pi: theta -= 2 * math.pi
    while theta < -math.pi: theta += 2 * math.pi

    if t - last_log_t > 0.1:
        log_writer.writerow([f"{t:.3f}", f"{x:.4f}", f"{y:.4f}",
                             phase, f"{goal_x:.4f}", f"{goal_y:.4f}"])
        log_file.flush()
        last_log_t = t

log_file.close()
print(f"[{name}] Stopped")