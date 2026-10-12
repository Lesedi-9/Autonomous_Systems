"""
my_controller.py
Component A - Step 4: PID path-following with odometry.
No GPS/Compass needed - the robot's position is estimated by
integrating the differential-drive kinematic model.
"""

from controller import Robot
import math
import csv

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
PATH_CSV = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\my_controller\path.csv"
TRAJ_CSV = r"C:\Users\User\OneDrive\Documents\AutonomousSystems\controllers\my_controller\actual_trajectory.csv"

Kp, Ki, Kd = 2.5, 0.4, 0.6
MAX_SPEED = 6.28
WHEEL_RADIUS = 0.0205
WHEELBASE = 0.052
LINEAR_SPEED = 0.10
WAYPOINT_TOL = 0.12

# Starting pose (must match Webots world: translation -2 -2 0, theta = 0)
X0, Y0, THETA0 = -2.0, -2.0, 0.0

# ------------------------------------------------------------------
# Load planned path
# ------------------------------------------------------------------
waypoints = []
with open(PATH_CSV) as f:
    reader = csv.reader(f); next(reader)
    for row in reader:
        waypoints.append((float(row[0]), float(row[1])))
print(f"Loaded {len(waypoints)} waypoints")

# ------------------------------------------------------------------
# Webots setup
# ------------------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def wrap_angle(a):
    while a > math.pi:  a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

# ------------------------------------------------------------------
# State
# ------------------------------------------------------------------
x, y, theta = X0, Y0, THETA0
integral = 0.0
prev_error = 0.0
stuck_timer = 0
stuck_last_wp = -1

log_file = open(TRAJ_CSV, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["t", "x", "y", "theta", "wp_index"])

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
wp_index = 0
t = 0.0
max_steps = 10000

while robot.step(timestep) != -1 and wp_index < len(waypoints) and max_steps > 0:
    max_steps -= 1
    t += dt

    tx, ty = waypoints[wp_index]
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)

    # Reached current waypoint?
    if dist < WAYPOINT_TOL:
        wp_index += 1
        integral = 0.0
        prev_error = 0.0
        stuck_timer = 0
        stuck_last_wp = wp_index
        if wp_index >= len(waypoints):
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            print("Goal reached.")
            break
        continue

    # Stuck detector
    if wp_index == stuck_last_wp:
        stuck_timer += 1
    else:
        stuck_timer = 0
        stuck_last_wp = wp_index

    if stuck_timer > 300:
        print(f"Skipping stuck waypoint {wp_index}")
        wp_index += 1
        stuck_timer = 0
        integral = 0.0
        prev_error = 0.0
        continue

    # PID on heading error
    desired_theta = math.atan2(dy, dx)
    error = wrap_angle(desired_theta - theta)

    integral += error * dt
    derivative = (error - prev_error) / dt
    omega = Kp * error + Ki * integral + Kd * derivative
    prev_error = error

    # Reduce forward speed when turning
    speed_factor = max(0.2, 1.0 - abs(error) / math.pi)
    v_forward = LINEAR_SPEED * speed_factor

    v_l = v_forward - omega * WHEELBASE / 2.0
    v_r = v_forward + omega * WHEELBASE / 2.0

    w_l = max(-MAX_SPEED, min(MAX_SPEED, v_l / WHEEL_RADIUS))
    w_r = max(-MAX_SPEED, min(MAX_SPEED, v_r / WHEEL_RADIUS))

    left_motor.setVelocity(w_l)
    right_motor.setVelocity(w_r)

    # ---- Odometry update (differential-drive kinematic model) ----
    v_l_actual = w_l * WHEEL_RADIUS
    v_r_actual = w_r * WHEEL_RADIUS
    v = (v_l_actual + v_r_actual) / 2.0
    omega_actual = (v_r_actual - v_l_actual) / WHEELBASE

    x     += v * math.cos(theta) * dt
    y     += v * math.sin(theta) * dt
    theta  = wrap_angle(theta + omega_actual * dt)

    # Log
    log_writer.writerow([f"{t:.3f}", f"{x:.4f}", f"{y:.4f}",
                         f"{theta:.4f}", wp_index])

    if int(t * 10) % 20 == 0:
        print(f"t={t:.1f}s  pos=({x:.2f},{y:.2f})  wp={wp_index}/{len(waypoints)}  err={error:.3f}")

log_file.close()
print(f"Trajectory saved to {TRAJ_CSV}")