================================================================
AUTONOMOUS SYSTEMS AND ROBOTICS 700
Intelligent Autonomous Robotic Systems Project
================================================================

Author:         Lesedi Magakwe
Student No:     402311859
Submission:     12 October 2026

----------------------------------------------------------------
1. OVERVIEW
----------------------------------------------------------------

This project contains two integrated autonomous robotic systems
built entirely in Webots and Python:

  Component A - Single-robot navigation with perception,
                Dijkstra/A* planning, and PID control.
  Component B - Multi-agent swarm with average consensus,
                PSO task allocation, and a trained neural network.

----------------------------------------------------------------
2. SOFTWARE REQUIREMENTS
----------------------------------------------------------------

  Webots:    R2025a (or later)
  Python:    3.14 (system) + a local virtual environment (.venv)
  OS tested: Windows 11

Python packages (installed into .venv):
  numpy
  matplotlib
  scikit-learn

To install:
  (in PyCharm terminal, with .venv active)
  pip install numpy matplotlib scikit-learn

Webots preference: Tools -> Preferences -> General
  Python command:  C:\Users\<user>\AppData\Local\Programs\Python\Python314\python.exe
  Startup mode:    Pause

----------------------------------------------------------------
3. FOLDER STRUCTURE
----------------------------------------------------------------

  worlds/
    component_a.wbt        - single-robot arena
    component_b.wbt        - three-robot swarm arena

  controllers/
    my_controller/         - Component A controller
    swarm_controller/      - Component B controller (shared by all 3 agents)

  scripts/
    occupancy_grid.py      - builds occupancy grid, runs Dijkstra + A*
    pid_tuning.py          - standalone PID tuning simulator
    plot_trajectory.py     - planned vs actual (Component A)
    plot_consensus.py      - consensus convergence (Component B)
    pso_allocation.py      - PSO task assignment
    plot_final.py          - swarm trajectory visualisation
    ml_assignment.py       - neural network training + evaluation

  figures/                 - exported PNGs for the report
  AutonomousSystems_Report.docx - the report
  README.txt               - this file

----------------------------------------------------------------
4. HOW TO RUN
----------------------------------------------------------------

COMPONENT A:

  Step 1. Run scripts/occupancy_grid.py in PyCharm.
          -> Generates path.csv in controllers/my_controller/

  Step 2. Open worlds/component_a.wbt in Webots.
          Press the "Run in Fast Mode" (double-triangle) button.
          Wait until the console prints "Goal reached."
          Reset the simulation.

  Step 3. Run scripts/plot_trajectory.py in PyCharm.
          -> Produces planned-vs-actual plot and metrics.

  For PID tuning curves: run scripts/pid_tuning.py.

COMPONENT B:

  Step 1. Run scripts/pso_allocation.py in PyCharm.
          -> Produces assignment.csv in controllers/swarm_controller/

  Step 2. Delete controllers/swarm_controller/comms/ (if it exists).

  Step 3. Open worlds/component_b.wbt in Webots.
          Press "Run in Fast Mode".
          Let it run ~35 seconds of simulated time.
          Reset the simulation.

  Step 4. Run scripts/plot_consensus.py and scripts/plot_final.py
          to visualise swarm behaviour.

  Step 5. Run scripts/ml_assignment.py to train and evaluate
          the neural network.

----------------------------------------------------------------
5. EXPECTED RUNTIME
----------------------------------------------------------------

  Component A Webots sim:      ~15 s (fast mode)
  Component B Webots sim:      ~35 s (fast mode)
  plot_trajectory.py:          ~2 s
  pso_allocation.py:           ~5 s
  ml_assignment.py:            ~60-90 s (includes learning curve)
  pid_tuning.py:               ~3 s

----------------------------------------------------------------
6. HARDWARE REQUIREMENTS
----------------------------------------------------------------

  Minimum: 4 GB RAM, dual-core CPU, OpenGL 3.3-capable GPU
  Recommended: 8 GB RAM, quad-core CPU, dedicated GPU

----------------------------------------------------------------
7. TROUBLESHOOTING
----------------------------------------------------------------

  "Cannot find controller file"
    - Ensure controllers/ is a sibling of worlds/ in the project.
    - Verify the controller file ends in .py (not .py.txt).

  "ModuleNotFoundError: numpy / matplotlib / sklearn"
    - Activate the .venv in PyCharm before running scripts.

  "PermissionError" on comms files
    - Delete comms/ folder and re-run Webots. This is a
      Windows file-locking race condition.

  Webots crashes on launch
    - Check Tools -> Preferences -> General -> Python command
      points at the real Python, not the WindowsApps stub.

================================================================
END OF README
================================================================
