"""
pid_tuning.py
Component A - Step 3: PID controller design and tuning.

Simulates a differential-drive robot trying to reach a target heading.
Compares three gain sets:
    - Untuned (Kp only)
    - Over-tuned (aggressive)
    - Tuned (balanced)
Plots heading response and error over time.
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Simulation parameters
# ------------------------------------------------------------------
dt = 0.01                # timestep (s)
T  = 8.0                 # total sim time (s)
steps = int(T / dt)

# Robot kinematics constants (e-puck)
L = 0.052                # wheelbase (m)
MAX_WHEEL_SPEED = 6.28   # rad/s
WHEEL_RADIUS = 0.0205    # m

# We'll simulate only the heading loop for clarity
# Target heading = 1.0 rad, starting at 0 rad
target_heading = 1.0
initial_heading = 0.0

# ------------------------------------------------------------------
# 2. PID controller class
# ------------------------------------------------------------------
class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return u

# ------------------------------------------------------------------
# 3. Simulate a heading loop for given gains
# ------------------------------------------------------------------
def simulate(Kp, Ki, Kd):
    pid = PID(Kp, Ki, Kd, dt)
    heading = initial_heading
    history_h = [heading]
    history_e = [target_heading - heading]
    history_t = [0.0]

    for i in range(1, steps):
        error = target_heading - heading
        omega = pid.compute(error)          # controller output = angular velocity
        heading += omega * dt               # simple first-order kinematics
        history_h.append(heading)
        history_e.append(error)
        history_t.append(i * dt)

    return np.array(history_t), np.array(history_h), np.array(history_e)

# ------------------------------------------------------------------
# 4. Three gain sets: untuned, over-tuned, tuned
# ------------------------------------------------------------------
configs = [
    ("Untuned (Kp=0.5, Ki=0, Kd=0)",    0.5, 0.0, 0.0,  'tab:red'),
    ("Over-tuned (Kp=6, Ki=2, Kd=0.05)", 6.0, 2.0, 0.05, 'tab:orange'),
    ("Tuned (Kp=2.5, Ki=0.4, Kd=0.6)",  2.5, 0.4, 0.6,  'tab:green'),
]

# ------------------------------------------------------------------
# 5. Compute performance metrics
# ------------------------------------------------------------------
def metrics(t, h, target):
    error = np.abs(target - h)
    # Rise time: first time reaching 90% of target
    thresh90 = 0.9 * target
    rise_idx = np.argmax(h >= thresh90) if np.any(h >= thresh90) else -1
    rise_time = t[rise_idx] if rise_idx >= 0 else np.nan

    # Overshoot
    overshoot = max(0.0, (h.max() - target) / target * 100)

    # Settling time (within 2% of target)
    band = 0.02 * target
    within = np.abs(h - target) <= band
    # Find last index outside the band, settle = next index
    outside = np.where(~within)[0]
    settle_time = t[outside[-1] + 1] if len(outside) > 0 and outside[-1] + 1 < len(t) else 0.0

    # Steady-state error (mean of last 10% of signal)
    n_tail = len(h) // 10
    sse = np.mean(np.abs(target - h[-n_tail:]))
    return rise_time, overshoot, settle_time, sse

# ------------------------------------------------------------------
# 6. Run all three simulations and plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

print("=" * 80)
print(f"{'Configuration':<35}{'Rise(s)':<10}{'Over(%)':<10}{'Settle(s)':<12}{'SSE':<10}")
print("=" * 80)

for label, Kp, Ki, Kd, color in configs:
    t, h, e = simulate(Kp, Ki, Kd)

    # Heading response
    axes[0].plot(t, h, label=label, color=color, linewidth=2)

    # Error response
    axes[1].plot(t, e, label=label, color=color, linewidth=2)

    rise, over, settle, sse = metrics(t, h, target_heading)
    print(f"{label:<35}{rise:<10.3f}{over:<10.2f}{settle:<12.3f}{sse:<10.4f}")

print("=" * 80)

# Reference line on heading plot
axes[0].axhline(target_heading, color='black', linestyle='--',
                linewidth=1, label=f'Target = {target_heading} rad')
axes[0].set_ylabel('Heading (rad)')
axes[0].set_title('Heading Response — Effect of PID Tuning')
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc='lower right', fontsize=9)

axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Heading Error (rad)')
axes[1].set_title('Heading Error — Effect of PID Tuning')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()