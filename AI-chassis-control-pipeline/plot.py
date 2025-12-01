import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# -------------------------------
# UDP Setup
# -------------------------------
UDP_IP = "0.0.0.0"
UDP_PORT = 1600

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# Latest IMU values
accel = np.array([0.0, 0.0, 0.0])
omega = np.array([0.0, 0.0, 0.0])

# -------------------------------
# Matplotlib Figure
# -------------------------------
fig = plt.figure(figsize=(12,6))

ax_acc = fig.add_subplot(121, projection='3d')
ax_omega = fig.add_subplot(122, projection='3d')

# Initial quiver objects
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

setup_3d(ax_acc, "Acceleration Vector (m/s²)")
setup_3d(ax_omega, "Angular Velocity Vector (deg/s)")

# -------------------------------
# Update loop
# -------------------------------
def update(frame):
    global accel, omega, acc_quiver, omega_quiver

    # Read all pending UDP packets
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except BlockingIOError:
            break

        try:
            fields = data.decode().split(",")
            if len(fields) < 11:
                continue

            # parse SARA signals (fields 2-7)
            accel_x = float(fields[2])
            accel_y = float(fields[3])
            accel_z = float(fields[4])

            omega_x = float(fields[5])
            omega_y = float(fields[6])
            omega_z = float(fields[7])

            accel = np.array([accel_x, accel_y, accel_z])
            omega = np.array([omega_x, omega_y, omega_z])

        except:
            continue

    # Remove old arrows
    if acc_quiver: 
        acc_quiver.remove()
    if omega_quiver: 
        omega_quiver.remove()

    # Draw new arrows
    acc_quiver = ax_acc.quiver(
        0, 0, 0,
        accel[0], accel[1], accel[2],
        length=1.0, normalize=False, color='blue'
    )

    omega_quiver = ax_omega.quiver(
        0, 0, 0,
        omega[0], omega[1], omega[2],
        length=1.0, normalize=False, color='red'
    )

    # Auto scale axes
    max_acc = max(50, np.linalg.norm(accel) * 1.5)
    max_omega = max(50, np.linalg.norm(omega) * 1.5)

    for ax, m in [(ax_acc, max_acc), (ax_omega, max_omega)]:
        ax.set_xlim([-m, m])
        ax.set_ylim([-m, m])
        ax.set_zlim([-m, m])

    return []

ani = FuncAnimation(fig, update, interval=50)
plt.tight_layout()
plt.show()
