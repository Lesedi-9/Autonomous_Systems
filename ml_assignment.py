"""
ml_assignment.py
Component B - Step 4: Neural network for optimal assignment prediction.

Improved version: distance-matrix features + feature scaling + more training.
Also identifies ties in the ground-truth labelling.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ------------------------------------------------------------------
# All possible assignments
# ------------------------------------------------------------------
PERMS = list(permutations(range(3)))
PERM_LABELS = [''.join(str(x) for x in p) for p in PERMS]
print(f"Classes: {PERM_LABELS}")

def assignment_cost(agents, targets, perm):
    """Total travel cost of a specific assignment."""
    return sum(
        np.hypot(agents[a][0] - targets[perm[a]][0],
                 agents[a][1] - targets[perm[a]][1])
        for a in range(3)
    )

def optimal_assignment(agents, targets):
    """Return the index of the first (lowest-cost) optimal assignment."""
    costs = [assignment_cost(agents, targets, p) for p in PERMS]
    return int(np.argmin(costs))

def distance_features(agents, targets):
    """9 features: distance from each agent to each target."""
    f = []
    for a in agents:
        for t in targets:
            f.append(np.hypot(a[0] - t[0], a[1] - t[1]))
    return f

# ------------------------------------------------------------------
# Generate dataset
# ------------------------------------------------------------------
N_SAMPLES = 5000
ARENA = 3.0
rng = np.random.default_rng(42)

X = np.zeros((N_SAMPLES, 9))
y = np.zeros(N_SAMPLES, dtype=int)

for i in range(N_SAMPLES):
    agents  = [(rng.uniform(-ARENA, ARENA), rng.uniform(-ARENA, ARENA)) for _ in range(3)]
    targets = [(rng.uniform(-ARENA, ARENA), rng.uniform(-ARENA, ARENA)) for _ in range(3)]
    X[i] = distance_features(agents, targets)
    y[i] = optimal_assignment(agents, targets)

print(f"Dataset: {N_SAMPLES} samples, 9 features")
print(f"Class distribution: {np.bincount(y)}")

# ------------------------------------------------------------------
# Split + scale features
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ------------------------------------------------------------------
# Train a larger MLP with early stopping
# ------------------------------------------------------------------
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42,
)
clf.fit(X_train_s, y_train)
print(f"Converged after {clf.n_iter_} iterations")

# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.3f}")

# ------------------------------------------------------------------
# World configuration test — with tie analysis
# ------------------------------------------------------------------
test_agents  = [(-2.5, -2.5), (2.5, -2.5), (0.0, 2.5)]
test_targets = [(-2.0, 1.5), (2.0, 1.5), (0.0, -2.0)]

print("\n--- World configuration analysis ---")
print("Cost of each assignment:")
costs = []
for i, perm in enumerate(PERMS):
    c = assignment_cost(test_agents, test_targets, perm)
    costs.append(c)
    print(f"  {PERM_LABELS[i]}: {c:.3f} m")

best = min(costs)
tied = [PERM_LABELS[i] for i, c in enumerate(costs) if abs(c - best) < 1e-6]
print(f"Optimal cost: {best:.3f} m")
print(f"Tied optima: {tied}")

# What the NN predicts
test_feat = np.array([distance_features(test_agents, test_targets)])
test_feat_s = scaler.transform(test_feat)
pred_class = clf.predict(test_feat_s)[0]
true_class = optimal_assignment(test_agents, test_targets)
pred_label = PERM_LABELS[pred_class]
true_label = PERM_LABELS[true_class]

print(f"\nNN predicted: {pred_label}")
print(f"Brute-force first optimum: {true_label}")
print(f"Exact string match: {pred_class == true_class}")
print(f"Both are optimal: {pred_label in tied and true_label in tied}")

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm = confusion_matrix(y_test, y_pred, labels=range(len(PERMS)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=PERM_LABELS)
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'Confusion Matrix (accuracy = {acc:.3f})')

train_sizes, train_scores, val_scores = learning_curve(
    clf, X_train_s, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 8), scoring='accuracy'
)
axes[1].plot(train_sizes, train_scores.mean(axis=1), 'o-',
             color='tab:blue', label='Training')
axes[1].plot(train_sizes, val_scores.mean(axis=1), 'o-',
             color='tab:orange', label='Validation (5-fold CV)')
axes[1].set_xlabel('Training set size')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Learning Curve')
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc='lower right')
axes[1].set_ylim(0.0, 1.05)

plt.tight_layout()
plt.show()