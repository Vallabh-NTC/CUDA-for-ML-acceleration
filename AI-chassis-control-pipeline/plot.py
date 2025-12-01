import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

UDP_IP = "0.0.0.0"
UDP_PORT = 1600

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

# ---- We plot only angle + speed for now ----
window_duration = 2.0
timestamps = deque()
angles = deque()
speeds = deque()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
line1, = ax1.plot([], [], label="Steering Angle")
line2, = ax2.plot([], [], label="Steering Speed")

ax1.set_ylabel("Angle (deg)")
ax2.set_ylabel("Speed (deg/s)")
ax2.set_xlabel("Time (s)")
ax1.grid(True)
ax2.grid(True)


def update(frame):
    while True:
        try:
            data, addr = sock.recvfrom(1024)
        except BlockingIOError:
            break

        try:
            text = data.decode().strip()

            # Expecting 10 CSV fields
            parts = text.split(",")

            if len(parts) != 10:
                print("Invalid CSV received:", text)
                continue

            angle = float(parts[0])
            speed = float(parts[1])

            ts = time.time()
            timestamps.append(ts)
            angles.append(angle)
            speeds.append(speed)

        except Exception as e:
            print("Parsing error:", e)

    # Remove old points
    now = time.time()
    while timestamps and now - timestamps[0] > window_duration:
        timestamps.popleft()
        angles.popleft()
        speeds.popleft()

    if timestamps:
        times = [t - timestamps[0] for t in timestamps]

        line1.set_data(times, angles)
        line2.set_data(times, speeds)

        ax1.set_xlim(0, max(0.1, times[-1]))
        ax2.set_xlim(0, max(0.1, times[-1]))

        ax1.set_ylim(min(angles) - 1, max(angles) + 1)
        ax2.set_ylim(min(speeds) - 1, max(speeds) + 1)

    return line1, line2


ani = FuncAnimation(fig, update, interval=50)
plt.tight_layout()
plt.show()
