import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, IMMEstimator
from filterpy.common import Q_discrete_white_noise, Saver
from math import sin, cos, radians, sqrt

# --- Configuration ---
dt = 0.1  # Time step
SIM_TIME = 60  # Simulation time in seconds
NUM_STEPS = int(SIM_TIME / dt)

# --- Motion Models Definitions ---

# 1. Constant Velocity (CV) Model
# State vector: [x, y, vx, vy]^T
# Measurement: [x, y]^T

cv_filter = KalmanFilter(dim_x=4, dim_z=2)
cv_filter.F = np.array([[1, 0, dt, 0],   # State Transition Matrix F
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
cv_filter.H = np.array([[1, 0, 0, 0],   # Measurement Function H
                        [0, 1, 0, 0]])
cv_filter.R = np.diag([0.25, 0.25])     # Measurement Noise R (std dev of 0.5 m)
# Process noise Q
q_cv = Q_discrete_white_noise(dim=2, dt=dt, var=0.05) # Process noise for velocity
cv_filter.Q = np.block([[np.zeros((2,2)), np.zeros((2,2))],
                        [np.zeros((2,2)), q_cv]])
cv_filter.P *= 100.                     # Initial covariance
cv_filter.x = np.array([[0, 0, 0, 0]]).T # Initial state

# 2. Coordinated Turn (CT) Model (using Extended Kalman Filter)
# State vector for EKF: [x, y, v, theta, omega]^T
# x, y: position
# v: speed
# theta: heading (angle from x-axis)
# omega: turn rate (rad/s)
# Measurement: [x, y]^T

class CoordinatedTurnEKF(ExtendedKalmanFilter):
    def __init__(self, dt, r_std, q_std_accel, q_std_turn_rate):
        # State: [x, y, v, heading, turn_rate]
        super().__init__(dim_x=5, dim_z=2)
        self.dt = dt
        self.R = np.diag([r_std**2, r_std**2])
        self.Q = np.diag([0, 0, (q_std_accel*dt)**2, 0, (q_std_turn_rate*dt)**2]) # Simplified Q
        self.P *= 100

    def fx(self, x, dt):
        """ State transition function """
        x_k, y_k, v_k, hdg_k, omega_k = x[:, 0] # Extract current state

        # Handle near-zero turn rate to avoid division by zero
        if abs(omega_k) < 1e-6: # Essentially straight
            x_next = x_k + v_k * dt * cos(hdg_k)
            y_next = y_k + v_k * dt * sin(hdg_k)
        else: # Turning
            x_next = x_k + (v_k / omega_k) * (sin(hdg_k + omega_k * dt) - sin(hdg_k))
            y_next = y_k + (v_k / omega_k) * (cos(hdg_k) - cos(hdg_k + omega_k * dt))

        v_next = v_k
        hdg_next = (hdg_k + omega_k * dt) % (2 * np.pi) # Normalize heading
        omega_next = omega_k

        return np.array([[x_next, y_next, v_next, hdg_next, omega_next]]).T

    def F_jacobian(self, x, dt):
        """ Jacobian of the state transition function fx """
        _x_k, _y_k, v_k, hdg_k, omega_k = x[:, 0]
        F = np.eye(5) # Initialize with identity

        if abs(omega_k) < 1e-6: # Straight motion
            F[0, 2] = dt * cos(hdg_k)
            F[0, 3] = -v_k * dt * sin(hdg_k)
            F[1, 2] = dt * sin(hdg_k)
            F[1, 3] = v_k * dt * cos(hdg_k)
        else: # Turning motion
            # dx/dv
            F[0, 2] = (sin(hdg_k + omega_k * dt) - sin(hdg_k)) / omega_k
            # dx/dhdg
            F[0, 3] = (v_k / omega_k) * (cos(hdg_k + omega_k * dt) - cos(hdg_k))
            # dx/domega
            term1_domega = - (v_k / omega_k**2) * (sin(hdg_k + omega_k * dt) - sin(hdg_k))
            term2_domega = (v_k / omega_k) * (dt * cos(hdg_k + omega_k * dt))
            F[0, 4] = term1_domega + term2_domega

            # dy/dv
            F[1, 2] = (cos(hdg_k) - cos(hdg_k + omega_k * dt)) / omega_k
            # dy/dhdg
            F[1, 3] = (v_k / omega_k) * (-sin(hdg_k) + sin(hdg_k + omega_k * dt))
            # dy/domega
            term1_domega_y = - (v_k / omega_k**2) * (cos(hdg_k) - cos(hdg_k + omega_k * dt))
            term2_domega_y = (v_k / omega_k) * (dt * sin(hdg_k + omega_k * dt))
            F[1, 4] = term1_domega_y + term2_domega_y

        F[3, 4] = dt # dhdg/domega
        return F

    def Hx(self, x):
        """ Measurement function (measures x, y) """
        return np.array([x[0,0], x[1,0]])

    def H_jacobian(self, x):
        """ Jacobian of the measurement function Hx """
        return np.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0]])

# Instantiate CT filter
ct_filter = CoordinatedTurnEKF(dt=dt, R_std=0.5, Q_std_accel=0.1, Q_std_turn_rate=0.03)
# Initial state for CT model [x, y, v, heading, omega]
ct_filter.x = np.array([[0, 0, 0, 0, 0]]).T # Start with 0 speed, 0 heading, 0 turn rate


# --- IMM Filter Setup ---
filters = [cv_filter, ct_filter]
mu = np.array([0.5, 0.5])  # Initial model probabilities (equal)
# Transition Probability Matrix (TPM) M_ij = P(model j at k | model i at k-1)
# Rows: from_model, Columns: to_model
#       CV   CT
# CV [[0.97, 0.03],  # From CV to CV, From CV to CT
# CT  [0.03, 0.97]]  # From CT to CV, From CT to CT
trans_prob_matrix = np.array([[0.97, 0.03],
                              [0.03, 0.97]])

imm = IMMEstimator(filters, mu, trans_prob_matrix)

# --- Data Generation ---
def generate_true_path_and_measurements(num_steps, dt):
    xs_true = []
    zs = []
    current_x = 0.0
    current_y = 0.0
    current_vx = 2.0  # m/s
    current_vy = 1.0  # m/s
    current_v = sqrt(current_vx**2 + current_vy**2)
    current_hdg = np.arctan2(current_vy, current_vx)
    current_omega = 0.0 # rad/s
    mode = 'CV'

    # Define measurement noise
    r_val = 0.5**2 # Measurement noise variance used in filters

    for i in range(num_steps):
        # Change mode at certain times
        if 100 <= i < 250: # Time steps for first turn
            mode = 'CT'
            current_omega = radians(15) # 15 deg/s turn rate
        elif 250 <= i < 350 : # Straight again
             mode = 'CV'
             current_omega = 0.0
        elif 350 <= i < 500: # Second turn, opposite direction
            mode = 'CT'
            current_omega = radians(-12) # -12 deg/s turn rate
        else: # Straight
            mode = 'CV'
            current_omega = 0.0


        if mode == 'CV':
            current_x += current_vx * dt
            current_y += current_vy * dt
            # velocities remain constant in CV for this simple sim
            # In reality, they would be updated by the CT model before switching to CV
        elif mode == 'CT':
            # Update heading and position based on turn
            if abs(current_omega) < 1e-6: # Straight
                current_x += current_v * dt * cos(current_hdg)
                current_y += current_v * dt * sin(current_hdg)
            else: # Turning
                prev_hdg = current_hdg
                current_hdg += current_omega * dt
                current_x += (current_v / current_omega) * (sin(current_hdg) - sin(prev_hdg))
                current_y += (current_v / current_omega) * (cos(prev_hdg) - cos(current_hdg))

            current_hdg %= (2 * np.pi)
            # Update vx, vy based on new heading and speed (v is constant in this CT model)
            current_vx = current_v * cos(current_hdg)
            current_vy = current_v * sin(current_hdg)


        xs_true.append([current_x, current_y, current_vx, current_vy, current_v, current_hdg, current_omega])
        # Add measurement noise
        z_x = current_x + np.random.randn() * sqrt(R_val)
        z_y = current_y + np.random.randn() * sqrt(R_val)
        zs.append(np.array([z_x, z_y]))

    return np.array(xs_true), np.array(zs)

xs_true, zs = generate_true_path_and_measurements(NUM_STEPS, dt)

# Set initial state for IMM based on first measurement or prior knowledge
# For CV filter (x, y, vx, vy) - estimate vx, vy from first few measurements if possible, or 0
imm.filters[0].x = np.array([[zs[0,0], zs[0,1], 0, 0]]).T
# For CT filter (x, y, v, hdg, omega)
initial_v = 0.1 # Small initial speed
initial_hdg = 0
if len(zs) > 1: # Estimate initial heading roughly
    dx = zs[1,0] - zs[0,0]
    dy = zs[1,1] - zs[0,1]
    initial_hdg = np.arctan2(dy, dx)
    initial_v = sqrt(dx**2 + dy**2) / dt
imm.filters[1].x = np.array([[zs[0,0], zs[0,1], initial_v if initial_v > 0 else 0.1, initial_hdg, 0]]).T

# --- Run Simulation ---
saver = Saver(imm)
for i, z in enumerate(zs):
    # The EKF in filterpy's IMM expects a 1D array or list for z
    z_input = np.array([z[0], z[1]])
    imm.predict()
    imm.update(z_input)
    saver.save()
saver.to_array()

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

# Plot 1: Path
axes[0].plot(xs_true[:, 0], xs_true[:, 1], 'k-', label='True Path')
axes[0].plot(zs[:, 0], zs[:, 1], 'r.', markersize=3, label='Measurements')
# Extract IMM estimated path. The IMM's x_post combines estimates.
# CV model's state is [x,y,vx,vy]. CT model's state is [x,y,v,hdg,omega].
# The IMM.x_post will be a weighted average. We need to ensure it's plotted correctly.
# If filterpy's IMM `x_post` is a mix, and the state vectors differ in meaning beyond first 2 elements,
# we might prefer to plot based on the most likely model or ensure a common part of state.
# Filterpy's IMM.x_post is the weighted mean of the bank of filters.
# We'll assume the first two elements [x,y] are consistent for plotting position.
axes[0].plot(saver.x_post[:, 0], saver.x_post[:, 1], 'b--', label='IMM Estimate')
axes[0].set_xlabel('X Position (m)')
axes[0].set_ylabel('Y Position (m)')
axes[0].legend()
axes[0].set_title('IMM Filter Tracking: Object Path')
axes[0].axis('equal')

# Plot 2: Model Probabilities
time_axis = np.arange(NUM_STEPS) * dt
axes[1].plot(time_axis, saver.mu[:, 0], label='P(CV Model)')
axes[1].plot(time_axis, saver.mu[:, 1], label='P(CT Model)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Model Probability')
axes[1].set_title('Model Probabilities Over Time')
axes[1].legend()
axes[1].set_ylim([-0.05, 1.05])

# Mark true mode changes for reference on probability plot
mode_changes = {'CV to CT (Turn 1)': 10.0, 'CT to CV (Straight)': 25.0, 'CV to CT (Turn 2)': 35.0, 'CT to CV (Straight)': 50.0}
for label, time_sec in mode_changes.items():
    axes[1].axvline(time_sec, color='gray', linestyle=':', lw=1)
    axes[1].text(time_sec + 0.5, 0.5, label.split('(')[0], rotation=90, verticalalignment='center', color='dimgray', fontsize=8)


plt.tight_layout()
plt.show()

print(f"Final state estimate (IMM): \n{imm.x_post.T}")
print(f"Final model probabilities: CV={imm.mu[0]:.3f}, CT={imm.mu[1]:.3f}")