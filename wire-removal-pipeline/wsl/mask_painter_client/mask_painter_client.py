#!/usr/bin/env python3
import argparse
import socket
import struct
import sys
import numpy as np
import cv2

# ---- protocol helpers --------------------------------------------------------
def fourcc_u32(tag: str) -> int:
    assert len(tag) == 4
    return struct.unpack("<I", tag.encode("ascii"))[0]

FRAME_MAGIC = fourcc_u32("FRAM")
MASK_MAGIC  = fourcc_u32("MASK")
MCNT_MAGIC  = fourcc_u32("MCNT")  # multi-mask count header

# FrameHeader: uint32 magic,width,height,stride,channels,size
FRAME_HDR_FMT = "<6I"
FRAME_HDR_SZ  = struct.calcsize(FRAME_HDR_FMT)

# MaskHeader: uint32 magic,width,height,size
MASK_HDR_FMT = "<4I"
MASK_HDR_SZ  = struct.calcsize(MASK_HDR_FMT)

# MasksCountHeader: uint32 magic,count
MCNT_HDR_FMT = "<2I"
MCNT_HDR_SZ  = struct.calcsize(MCNT_HDR_FMT)

def recv_all(sock: socket.socket, n: int) -> bytes:
    chunks = []
    got = 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise RuntimeError("socket closed while receiving")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)

def send_all(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        n = sock.send(view[sent:])
        if n <= 0:
            raise RuntimeError("socket send failed")
        sent += n

# ---- UI: simple polyline multi-mask (press 'n' to start a new mask) ----------
class PolylineMaskPainter:
    """
    Click-to-draw polylines (no freehand):
      - First click anchors; each next click draws a straight segment from previous→current.
      - Press 'n' to finish the current mask and start a new one.

    Size entry:
      - Type digits to enter a size (e.g., '12'), Enter to apply, Backspace to edit, Esc to cancel.
      - While editing size, Enter won't finish the session; it applies the size.

    Other keys:
      n: new mask  |  u: undo last segment
      [ ] or - + : brush size -/+
      r: reset current  |  i: invert current  |  p: preview toggle
      Enter/Space: accept (when NOT editing size)
      q/Esc: cancel (when NOT editing size)
    """
    def __init__(self, bgr_img: np.ndarray, brush: int = 9):
        self.img = bgr_img
        self.h, self.w = bgr_img.shape[:2]

        self.current_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.segments = []   # list[(p0, p1, brush)]
        self.anchor_pt = None
        self.brush = max(1, int(brush))
        self.preview = True

        self.masks = []  # committed masks (list of np.uint8 HxW)

        self.done = False
        self.cancel = False

        # numeric size entry state
        self.size_editing = False
        self.size_buffer = ""  # digits being typed

    def _rebuild_current(self):
        self.current_mask.fill(0)
        for p0, p1, b in self.segments:
            cv2.line(self.current_mask, p0, p1, 255, b, cv2.LINE_AA)

    def _commit_current(self):
        if np.any(self.current_mask):
            self.masks.append(self.current_mask.copy())
            self.segments.clear()
            self.current_mask.fill(0)
            self.anchor_pt = None
            return True
        return False

    # mouse
    def on_mouse(self, event, x, y, flags, param):
        x = int(np.clip(x, 0, self.w - 1)); y = int(np.clip(y, 0, self.h - 1))
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.anchor_pt is None:
                self.anchor_pt = pt
            else:
                p0, p1 = self.anchor_pt, pt
                cv2.line(self.current_mask, p0, p1, 255, self.brush, cv2.LINE_AA)
                self.segments.append((p0, p1, int(self.brush)))
                self.anchor_pt = pt
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.anchor_pt = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            # mouse wheel sizing (if supported by your OpenCV build)
            delta = 1 if (flags >> 16) > 0 else -1
            self.brush = int(np.clip(self.brush + delta, 1, 1024))
            print(f"[ui] brush = {self.brush}")

    def make_preview(self) -> np.ndarray:
        vis = self.img.copy()

        # committed masks in blue
        if self.masks:
            stacked = np.any(np.stack([(m > 0) for m in self.masks], axis=0), axis=0)
            blue = np.full_like(vis, (255, 0, 0))
            blended_b = cv2.addWeighted(vis, 1.0, blue, 0.30, 0)
            vis[stacked] = blended_b[stacked]

        # current mask in green
        if self.preview and np.any(self.current_mask):
            green = np.full_like(vis, (0, 255, 0))
            blended_g = cv2.addWeighted(vis, 1.0, green, 0.35, 0)
            m = self.current_mask > 0
            vis[m] = blended_g[m]

        if self.size_editing:
            hud = (f"Brush[{self.brush}]  Masks:{len(self.masks)}  "
                   f"[SIZE: {self.size_buffer or '_'}  (Enter apply • Backspace edit • Esc cancel)]")
        else:
            hud = (f"Brush[{self.brush}]  Masks:{len(self.masks)}  "
                   "[Click=segment | R-click clear anchor | "
                   "n new mask | u undo | [ / ] / - / + size | type digits to set size | "
                   "r reset | i invert | p prev | Enter=OK | q/Esc=cancel]")

        cv2.putText(vis, hud, (10, max(20, self.h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1, cv2.LINE_AA)
        return vis

    def _handle_size_key(self, key: int) -> bool:
        """
        Returns True if the key was handled in size-editing mode.
        """
        # digits
        if ord('0') <= key <= ord('9'):
            # avoid leading zeros unless zero is the only digit
            if key == ord('0') and self.size_buffer == "":
                self.size_buffer = "0"
            else:
                self.size_buffer += chr(key)
            return True

        # Backspace (8 in some builds, 127 in others)
        if key in (8, 127):
            self.size_buffer = self.size_buffer[:-1]
            return True

        # Enter: apply size and exit size-editing
        if key in (13, 10):
            if self.size_buffer:
                try:
                    val = int(self.size_buffer)
                    self.brush = int(np.clip(val, 1, 1024))
                    print(f"[ui] brush = {self.brush}")
                except ValueError:
                    pass
            self.size_editing = False
            self.size_buffer = ""
            return True

        # Esc: cancel size-editing
        if key == 27:
            self.size_editing = False
            self.size_buffer = ""
            return True

        return False

    def run(self, win_name="Mask Painter"):
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, self.on_mouse)

        while True:
            cv2.imshow(win_name, self.make_preview())
            key = cv2.waitKey(16) & 0xFF
            if key == 255:
                continue

            # If we are editing size, capture keys for that first.
            if self.size_editing:
                if self._handle_size_key(key):
                    continue  # processed inside size editor
                # if not handled, fall through (but Enter/Esc are already handled above)

            # Start size entry if a digit is typed (and we weren't already editing)
            if not self.size_editing and (ord('0') <= key <= ord('9')):
                self.size_editing = True
                self.size_buffer = chr(key)
                continue

            # Global keys (when not in size-edit mode)
            if not self.size_editing:
                if key in (ord('q'), 27):  # q or Esc
                    self.cancel = True
                    break
                if key in (13, 10, 32):    # Enter/Return/Space
                    self.done = True
                    break

            # Drawing/session controls (work in both modes except Enter/Esc are intercepted)
            if key == ord('n'):
                committed = self._commit_current()
                if committed:
                    print(f"[ui] new mask started (total committed: {len(self.masks)})")
                else:
                    print("[ui] current mask empty; nothing committed")

            elif key == ord('u'):
                if self.segments:
                    self.segments.pop()
                    self._rebuild_current()
                else:
                    print("[ui] nothing to undo")

            elif key in (ord('['), ord('-'), ord('_')):
                self.brush = max(1, self.brush - 1)
                print(f"[ui] brush = {self.brush}")

            elif key in (ord(']'), ord('='), ord('+')):
                self.brush = min(1024, self.brush + 1)
                print(f"[ui] brush = {self.brush}")

            elif key == ord('r'):
                self.segments.clear()
                self.current_mask.fill(0)
                self.anchor_pt = None

            elif key == ord('i'):
                cv2.bitwise_not(self.current_mask, self.current_mask)

            elif key == ord('p'):
                self.preview = not self.preview

        cv2.destroyWindow(win_name)
        # auto-commit on accept
        if self.done and np.any(self.current_mask):
            self.masks.append(self.current_mask.copy())
        return (not self.cancel) and self.done, self.masks

# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Client: receive frame, draw multiple masks ('n' = new mask), send them.")
    ap.add_argument("host", help="server host/IP")
    ap.add_argument("port", type=int, help="server port")
    ap.add_argument("--brush", type=int, default=9, help="initial brush size in pixels")
    args = ap.parse_args()

    addr = (args.host, args.port)
    with socket.create_connection(addr) as s:
        # 1) receive FrameHeader
        hdr_bytes = recv_all(s, FRAME_HDR_SZ)
        magic, w, h, stride, channels, size = struct.unpack(FRAME_HDR_FMT, hdr_bytes)
        if magic != FRAME_MAGIC:
            raise RuntimeError(f"Bad frame magic: got {magic:#x}")

        expected_stride = w * 4
        if channels != 4:
            raise RuntimeError(f"Expected 4 channels (RGBA), got {channels}")
        if stride != expected_stride:
            raise RuntimeError(f"Unexpected stride {stride}, expected {expected_stride}")
        if size != h * stride:
            raise RuntimeError(f"Unexpected frame size {size}, expected {h*stride}")

        # 2) RGBA payload → BGR
        rgba = np.frombuffer(recv_all(s, size), dtype=np.uint8)
        rgba = rgba.reshape((h, stride))[:, :w*4].reshape((h, w, 4))
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        # 3) UI
        painter = PolylineMaskPainter(bgr, brush=args.brush)
        ok, masks = painter.run()

        if not ok:
            print("Canceled by user; sending one empty mask.", file=sys.stderr)
            masks = [np.zeros((w*h,), dtype=np.uint8).reshape((h, w))]

        # 4) send output (ALWAYS multi-mask)
        count = len(masks)
        send_all(s, struct.pack(MCNT_HDR_FMT, MCNT_MAGIC, count))
        for idx, m in enumerate(masks, 1):
            mbytes = ((m > 0).astype(np.uint8) * 255).tobytes(order="C")
            mask_size = w * h
            mhdr = struct.pack(MASK_HDR_FMT, MASK_MAGIC, w, h, mask_size)
            send_all(s, mhdr)
            send_all(s, mbytes)
            print(f"[client] sent mask {idx}/{count}")
        print(f"[client] sent {count} masks")

    print("[client] done")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[client] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
