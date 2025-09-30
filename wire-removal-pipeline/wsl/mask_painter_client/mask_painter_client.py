#!/usr/bin/env python3
import argparse, socket, struct, sys, math
import numpy as np, cv2

# ---- Protocol ---------------------------------------------------------
def fourcc_u32(tag: str) -> int:
    return struct.unpack("<I", tag.encode("ascii"))[0]

FRAME_MAGIC = fourcc_u32("FRAM")
FRAME_HDR_FMT = "<6I"; FRAME_HDR_SZ = struct.calcsize(FRAME_HDR_FMT)

SCNT_MAGIC = fourcc_u32("SCNT")
STMP_MAGIC = fourcc_u32("STMP")
SCNT_HDR_FMT = "<2I"
STMP_FMT = "<I4f2I"

def recv_all(s, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = s.recv_into(view[got:])
        if r == 0:
            raise RuntimeError("socket closed while receiving")
        got += r
    return bytes(buf)

def send_all(s, data):
    view = memoryview(data)
    sent = 0
    while sent < len(view):
        n = s.send(view[sent:])
        if n <= 0:
            raise RuntimeError("socket send failed")
        sent += n

def clamp(v, a, b): return a if v < a else (b if v > b else v)
def odd(n): return int(n) | 1

# ---- Designer ---------------------------------------------------------
class StampDesigner:
    """
    Click-to-polyline authoring (rotating squares, edge-to-edge):

      Left-click: add a point.
        - 1st click: starts a wire (places first square at the point)
        - 2nd click: fills segment p1->p2 with attached squares
        - 3rd click: fills p2->p3, etc.
      Right-click / Backspace: undo last segment

    Hotkeys:
      a/d: rotate donor axis (−/+ 3°)
      w/s: donor separation ( +/− 2 px )
      [ / ]: block (square side) −/+ (kept odd)
      h: toggle HARD PASTE vs. alpha (preview text only)
      , / .: alpha −/+  (preview text only)
      m: toggle magnifier   z/x: magnifier scale −/+
      n: next wire (start fresh, keep params)
      f: next frame
      q/ESC: quit
    """
    def __init__(self, bgr_img, frame_idx=1, total_frames=1,
                 init_angle=90.0, init_sep=14, init_block=21, alpha=0.65,
                 mag_pad=50, mag_scale=6, mag_max=480,
                 step_px=None):  # step_px ignored for attachment; kept for CLI compat
        self.frame=bgr_img; self.h,self.w=bgr_img.shape[:2]
        self.cursor=(self.w//2,self.h//2)
        self.angle=float(init_angle); self.sep=float(init_sep)
        self.block=odd(int(init_block)); self.alpha=float(alpha)
        self.hard_paste=False

        self.wires=[]   # each wire: dict(points, seg_stack, stamps, last (dict) or None, rem(float))
        self._start_new_wire()

        # Magnifier
        self.mag_on=True; self.mag_pad=int(mag_pad)
        self.mag_scale=int(mag_scale); self.mag_max=int(mag_max)

        self.action="next_frame"
        self.frame_idx=frame_idx; self.total_frames=total_frames

        self.win="Stamp Designer"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, min(self.w,1280), min(self.h,720))
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ---- wire utilities ----
    def _start_new_wire(self):
        self.wires.append(dict(points=[], seg_stack=[], stamps=[], last=None, rem=0.0))
        self.cur = self.wires[-1]

    def _all_stamps(self):
        out=[]
        for w in self.wires:
            out.extend(w["stamps"])
        return out

    # ---- mouse ----
    def _on_mouse(self, event,x,y,flags,ud=None):
        x=int(clamp(x,0,self.w-1)); y=int(clamp(y,0,self.h-1))
        self.cursor=(x,y)
        if event==cv2.EVENT_LBUTTONDOWN:
            self._add_point_and_maybe_segment(x,y)
        elif event==cv2.EVENT_RBUTTONDOWN:
            self._undo_last_segment()

    # ---- geometry: exact tangent spacing for rotating squares ----
    @staticmethod
    def _half_extent_along_dir(side, sq_angle_deg, dir_phi_rad):
        # h(φ) = (S/2) * (|cos(φ-θ)| + |sin(φ-θ)|)
        S = float(side)
        delta = dir_phi_rad - math.radians(float(sq_angle_deg))
        return 0.5*S*(abs(math.cos(delta)) + abs(math.sin(delta)))

    @staticmethod
    def _delta_between(prev_side, prev_angle_deg, cur_side, cur_angle_deg, dir_phi_rad):
        return (StampDesigner._half_extent_along_dir(prev_side, prev_angle_deg, dir_phi_rad) +
                StampDesigner._half_extent_along_dir(cur_side,  cur_angle_deg,  dir_phi_rad))

    # ---- stamping primitives ----
    def _commit_stamp(self,x,y, side=None, ang=None, sep=None):
        if side is None: side = odd(self.block)
        if ang  is None: ang  = float(self.angle)
        if sep  is None: sep  = float(self.sep)
        st = dict(cx=float(x), cy=float(y),
                  angle_deg=float(ang),
                  sep_px=float(sep),
                  bw=int(side), bh=int(side))
        self.cur["stamps"].append(st)
        self.cur["last"] = dict(cx=st["cx"], cy=st["cy"], side=st["bw"], ang=st["angle_deg"])

    def _stamps_along_segment(self, x0,y0, x1,y1):
        dx=x1-x0; dy=y1-y0
        L=math.hypot(dx,dy)
        if L<=1e-6: return 0

        ux, uy = dx/L, dy/L
        dir_phi = math.atan2(uy, ux)
        side_now = odd(self.block)
        ang_now  = float(self.angle)

        start_count = len(self.cur["stamps"])
        d = 0.0

        # If no previous stamp in this wire, start by placing at the start point.
        if self.cur["last"] is None:
            self._commit_stamp(x0, y0, side_now, ang_now)
            # Next gap needed from this first to the next:
            self.cur["rem"] = self._delta_between(side_now, ang_now, side_now, ang_now, dir_phi)
            d = 0.0
        # Otherwise, we already carry a remaining gap from the previous segment in self.cur["rem"].

        # Walk forward along the segment, placing whenever remaining gap fits
        rem = float(self.cur["rem"])
        # distance already advanced from segment start: d
        while d + rem <= L + 1e-6:
            d += rem
            xs = x0 + ux * d
            ys = y0 + uy * d
            # Place with *current* size/angle
            self._commit_stamp(xs, ys, side_now, ang_now)
            # For the next placement, compute new required gap between just-placed and the next (same params)
            rem = self._delta_between(side_now, ang_now, side_now, ang_now, dir_phi)

        # Carry residual for next segment
        self.cur["rem"] = rem - (L - d)

        end_count = len(self.cur["stamps"])
        self.cur["seg_stack"].append((start_count, end_count))
        return (end_count - start_count)

    def _add_point_and_maybe_segment(self,x,y):
        pts=self.cur["points"]
        if not pts:
            pts.append((x,y))
            # If this is the very first point of the wire, place the first stamp immediately.
            side_now = odd(self.block)
            ang_now  = float(self.angle)
            self._commit_stamp(x, y, side_now, ang_now)
            # Prepare remaining distance for continuation on the (future) first segment.
            # We don't know its direction yet; when the next point arrives we’ll recompute rem using that dir.
            self.cur["rem"] = 0.0
        else:
            # When starting a new segment, if there was a last stamp but rem is zero (first click after start),
            # recompute the initial required gap using the new segment direction.
            x0,y0=pts[-1]
            pts.append((x,y))
            if self.cur["last"] is not None and abs(self.cur["rem"]) < 1e-6:
                side_now = odd(self.block)
                ang_now  = float(self.angle)
                dir_phi  = math.atan2(y - y0, x - x0)
                prev = self.cur["last"]
                self.cur["rem"] = self._delta_between(prev["side"], prev["ang"], side_now, ang_now, dir_phi)
            self._stamps_along_segment(x0,y0,x,y)

    def _undo_last_segment(self):
        if not self.cur["seg_stack"]:
            if self.cur["points"]:
                self.cur["points"].pop()
            return
        start,end=self.cur["seg_stack"].pop()
        # delete stamps from that segment
        del self.cur["stamps"][start:end]
        # remove last point
        if self.cur["points"]:
            self.cur["points"].pop()
        # rebuild last and rem from remaining stamps conservatively
        if self.cur["stamps"]:
            last = self.cur["stamps"][-1]
            self.cur["last"] = dict(cx=last["cx"], cy=last["cy"], side=last["bw"], ang=last["angle_deg"])
            self.cur["rem"] = 0.0  # will be recomputed when the next segment starts
        else:
            self.cur["last"] = None
            self.cur["rem"] = 0.0

    # ---- drawing helpers ----
    @staticmethod
    def _rot_square_corners(cx, cy, side, angle_deg):
        s = side * 0.5
        local = [(-s,-s),(+s,-s),(+s,+s),(-s,+s)]
        ang = math.radians(angle_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        pts = []
        for (x,y) in local:
            xr = x*ca - y*sa
            yr = x*sa + y*ca
            pts.append((int(round(cx + xr)), int(round(cy + yr))))
        return pts

    def _draw_rot_square(self, img, cx, cy, side, angle_deg, color, thickness=1):
        pts = self._rot_square_corners(cx, cy, side, angle_deg)
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i+1)%4]
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)

    # ---- drawing ----
    def _draw_preview(self, img):
        out=img.copy()

        # draw existing wires (rotated squares + polyline)
        for wi,w in enumerate(self.wires):
            col = (0,200,255) if (w is self.cur) else (120,120,120)
            for st in w["stamps"]:
                scx,scy=int(round(st["cx"])),int(round(st["cy"]))
                side = int(st["bw"])
                ang  = float(st["angle_deg"])
                self._draw_rot_square(out, scx, scy, side, ang, col, 1)

            pts=w["points"]
            for i,p in enumerate(pts):
                cv2.circle(out,(int(p[0]),int(p[1])),3,(0,255,0),-1)
                if i>0:
                    cv2.line(out,(int(pts[i-1][0]),int(pts[i-1][1])),(int(p[0]),int(p[1])),(0,255,0),1,cv2.LINE_AA)

        # live donor-axis preview + rotated square at cursor
        cx, cy = self.cursor
        ang = self.angle
        rad = math.radians(ang)
        vx, vy = math.cos(rad), math.sin(rad)
        half = self.sep / 2.0
        xt = cx - vx*half; yt = cy - vy*half
        xb = cx + vx*half; yb = cy + vy*half
        cv2.circle(out,(int(round(xt)),int(round(yt))),5,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(out,(int(round(xb)),int(round(yb))),5,(0,255,0),2,cv2.LINE_AA)

        side = odd(self.block)
        self._draw_rot_square(out, int(cx), int(cy), side, ang, (60,255,60), 1)

        # preview tail -> cursor
        pts=self.cur["points"]
        if pts:
            tail=pts[-1]
            cv2.line(out,(int(tail[0]),int(tail[1])),(int(self.cursor[0]),int(self.cursor[1])),(60,255,60),1,cv2.LINE_AA)

        # HUD
        total_stamps = sum(len(w["stamps"]) for w in self.wires)
        cur_stamps = len(self.cur["stamps"])
        hud=f"[F {self.frame_idx}/{self.total_frames}] [W {len(self.wires)}] angle:{self.angle:.1f}° sep:{self.sep:.1f}px block:{side} " \
            f"{'HARD' if self.hard_paste or self.alpha>=0.999 else f'alpha:{int(self.alpha*100)}%'} " \
            f"wire_stamps:{cur_stamps} total:{total_stamps} attached"
        cv2.putText(out,hud,(10,max(20,self.h-10)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

        # magnifier bbox around rotated square
        bbox_pts = self._rot_square_corners(cx, cy, side, ang)
        xs = [p[0] for p in bbox_pts]; ys = [p[1] for p in bbox_pts]
        x0 = clamp(min(xs), 0, self.w-1); y0 = clamp(min(ys), 0, self.h-1)
        x1 = clamp(max(xs)+1, 1, self.w); y1 = clamp(max(ys)+1, 1, self.h)
        self._add_magnifier(out,out,(int(x0),int(y0),int(x1),int(y1)))
        return out

    def _add_magnifier(self, canvas, preview_img, rect):
        if not self.mag_on: return
        x0,y0,x1,y1=rect
        x0=clamp(x0-self.mag_pad,0,self.w-1); y0=clamp(y0-self.mag_pad,0,self.h-1)
        x1=clamp(x1+self.mag_pad,1,self.w);   y1=clamp(y1+self.mag_pad,1,self.h)
        if x1<=x0 or y1<=y0: return
        crop=preview_img[int(y0):int(y1), int(x0):int(x1)]
        ch,cw=crop.shape[:2]
        scale=int(clamp(self.mag_scale,2,16))
        up_w=cw*scale; up_h=ch*scale
        if up_w>self.mag_max or up_h>self.mag_max:
            k=min(self.mag_max/up_w, self.mag_max/up_h)
            up_w=max(1,int(round(up_w*k))); up_h=max(1,int(round(up_h*k)))
        mag=cv2.resize(crop,(up_w,up_h),interpolation=cv2.INTER_NEAREST)
        margin=10; xa=canvas.shape[1]-up_w-margin; ya=margin
        bg=canvas.copy()
        cv2.rectangle(bg,(xa-4,ya-24),(xa+up_w+4,ya+up_h+4),(0,0,0),-1)
        cv2.addWeighted(bg,0.45,canvas,0.55,0,canvas)
        canvas[ya:ya+up_h, xa:xa+up_w]=mag
        cv2.rectangle(canvas,(xa-1,ya-1),(xa+up_w,ya+up_h),(255,255,255),1)
        cv2.putText(canvas,"magnifier",(xa,ya-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # ---- main loop ----
    def run(self):
        while True:
            cv2.imshow(self.win,self._draw_preview(self.frame))
            k=cv2.waitKey(16)&0xFF
            if k==255: continue
            if k in (ord('q'),27): self.action="quit"; break
            elif k==ord('f'): self.action="next_frame"; break
            elif k==ord('n'):
                if self.cur["points"] or self.cur["stamps"]:
                    self._start_new_wire()
            elif k==8 or k==ord('\b'):
                self._undo_last_segment()
            elif k==ord('h'):
                self.hard_paste = not self.hard_paste
            elif k==ord('a'): self.angle-=3.0
            elif k==ord('d'): self.angle+=3.0
            elif k==ord('w'): self.sep=clamp(self.sep+2,2,1000)
            elif k==ord('s'): self.sep=clamp(self.sep-2,2,1000)
            elif k==ord('['): self.block=clamp(odd(self.block-2),5,999)
            elif k==ord(']'): self.block=clamp(odd(self.block+2),5,999)
            elif k==ord('m'): self.mag_on = not self.mag_on
            elif k==ord('z'): self.mag_scale=int(clamp(self.mag_scale-1,2,16))
            elif k==ord('x'): self.mag_scale=int(clamp(self.mag_scale+1,2,16))
            elif k==ord(','): self.alpha=max(0.0, self.alpha-0.05)
            elif k==ord('.'): self.alpha=min(1.0, self.alpha+0.05)
            if self.angle<0: self.angle+=360.0
            if self.angle>=360.0: self.angle-=360.0
        cv2.destroyWindow(self.win)
        return self._all_stamps(), (self.action=="quit")

# ---- Frame loop -------------------------------------------------------
def author_one_frame(host, port, frame_idx, total_frames, args):
    addr=(host,port)
    with socket.create_connection(addr) as s:
        hdr=recv_all(s, FRAME_HDR_SZ)
        magic,w,h,stride,channels,size=struct.unpack(FRAME_HDR_FMT,hdr)
        if magic!=FRAME_MAGIC or channels!=4 or stride!=w*4 or size!=h*stride:
            raise RuntimeError("Bad frame header")
        rgba=np.frombuffer(recv_all(s,size),dtype=np.uint8)
        rgba=rgba.reshape((h,stride))[:,:w*4].reshape((h,w,4))
        bgr=cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

        designer=StampDesigner(
            bgr, frame_idx, total_frames,
            args.angle,args.sep,args.block,args.alpha,
            args.magpad,args.magscale,args.magmax
        )
        stamps,quit_all=designer.run()

        # send stamps
        send_all(s, struct.pack(SCNT_HDR_FMT, SCNT_MAGIC, len(stamps)))
        for st in stamps:
            pkt=struct.pack(STMP_FMT,STMP_MAGIC,
                            float(st["cx"]),float(st["cy"]),
                            float(st["angle_deg"]),float(st["sep_px"]),
                            int(st["bw"]),int(st["bh"]))
            send_all(s, pkt)
        print(f"[client] sent {len(stamps)} stamp(s) for frame {frame_idx}")
        return quit_all

def main():
    ap=argparse.ArgumentParser(description="Interactive polyline stamp authoring client (rotating squares, edge-to-edge).")
    ap.add_argument("host"); ap.add_argument("port",type=int)
    ap.add_argument("--frames",type=int,default=1)
    ap.add_argument("--angle",type=float,default=90.0)
    ap.add_argument("--sep",type=float,default=14.0)
    ap.add_argument("--block",type=int,default=21, help="square side length (odd)")
    ap.add_argument("--alpha",type=float,default=0.65)
    ap.add_argument("--magpad",type=int,default=50)
    ap.add_argument("--magscale",type=int,default=6)
    ap.add_argument("--magmax",type=int,default=480)
    args=ap.parse_args()
    args.block = odd(args.block)

    for i in range(args.frames):
        quit_all=author_one_frame(args.host,args.port,i+1,args.frames,args)
        if quit_all: break
    print("[client] done")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"[client] ERROR: {e}", file=sys.stderr); sys.exit(1)
