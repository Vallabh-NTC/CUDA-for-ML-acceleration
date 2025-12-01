import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import numpy as np
import time
import matplotlib.gridspec as gridspec

# ============================================================
# UDP SETUP
# ============================================================
UDP_IP = "0.0.0.0"
UDP_PORT = 1600

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# ============================================================
# BUFFERS FOR TIME-SERIES
# ============================================================
WINDOW = 2.0  # seconds

timestamps = deque()
steer_angle = deque()
steer_speed = deque()
gas_pedal = deque()
brake_pedal = deque()

# Latest IMU vectors
accel = np.zeros(3)
omega = np.zeros(3)

# ============================================================
# FIGURE LAYOUT — 3 LEFT PLOTS + 2 RIGHT 3D PLOTS
# ============================================================
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

# Left column (time-series)
ax1 = fig.add_subplot(gs[0, 0])   # steering + speed
ax2 = fig.add_subplot(gs[1, 0])   # gas pedal
ax3 = fig.add_subplot(gs[2, 0])   # brake pedal

# Right column (3D IMU)
ax_acc  = fig.add_subplot(gs[0:2, 1], projection='3d')  # tall 3D plot
ax_omega = fig.add_subplot(gs[2, 1], projection='3d')   # bottom 3D plot

# Lines for time-series
line_angle, = ax1.plot([], [], label="Steering (deg)")
line_speed, = ax1.plot([], [], label="Speed (deg/s)")
line_gas,   = ax2.plot([], [], label="Gas Pedal %")
line_brake, = ax3.plot([], [], label="Brake Pedal %")

ax1.legend()
ax2.legend()
ax3.legend()

for ax in [ax1, ax2, ax3]:
    ax.grid(True)

# 3D vector arrows
acc_quiver = None
omega_quiver = None

def setup_3d(ax, title):
    ax.set_title(title)
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

setup_3d(ax_acc, "Acceleration Vector")
setup_3d(ax_omega, "Angular Velocity Vector")

# ============================================================
# UPDATE LOOP
# ============================================================
def update(frame):
    global accel, omega, acc_quiver, omega_quiver

    # ----------------------
    # Read all available UDP packets
    # ----------------------
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except BlockingIOError:
            break

        try:
            f = data.decode().split(",")
            if len(f) < 12:
                continue

            ts = time.time()
            timestamps.append(ts)

            # CSV fields
            steer_angle.append(float(f[0]))
            steer_speed.append(float(f[1]))

            # IMU accel + omega
            accel = np.array([float(f[2]), float(f[3]), float(f[4])])
            omega = np.array([float(f[5]), float(f[6]), float(f[7])])

            # Pedals
            gas_pedal.append(float(f[10]))      # <-- corrected index
            brake_pedal.append(float(f[11]))

        except:
            continue

    # ---------------------------------------
    # Trim window
    # ---------------------------------------
    now = time.time()
    while timestamps and now - timestamps[0] > WINDOW:
        timestamps.popleft()
        steer_angle.popleft()
        steer_speed.popleft()
        gas_pedal.popleft()
        brake_pedal.popleft()

    # ---------------------------------------
    # Update time-series graphs
    # ---------------------------------------
    if timestamps:
        t = np.array(timestamps) - timestamps[0]

        line_angle.set_data(t, steer_angle)
        line_speed.set_data(t, steer_speed)
        line_gas.set_data(t, gas_pedal)
        line_brake.set_data(t, brake_pedal)

        # Autoscale X
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, max(0.1, t[-1]))

        # Autoscale Y
        ax1.set_ylim(
            min(steer_angle + steer_speed) - 1,
            max(steer_angle + steer_speed) + 1)

        ax2.set_ylim(min(gas_pedal) - 1, max(gas_pedal) + 1)
        ax3.set_ylim(min(brake_pedal) - 1, max(brake_pedal) + 1)

    # ---------------------------------------
    # Update 3D vectors
    # ---------------------------------------
    if acc_quiver:
        acc_quiver.remove()
    if omega_quiver:
        omega_quiver.remove()

    acc_quiver = ax_acc.quiver(0, 0, 0, accel[0], accel[1], accel[2], color="blue")
    omega_quiver = ax_omega.quiver(0, 0, 0, omega[0], omega[1], omega[2], color="red")

    return []

# ============================================================
# Run animation
# ============================================================
ani = FuncAnimation(fig, update, interval=50)
plt.tight_layout()
plt.show()
