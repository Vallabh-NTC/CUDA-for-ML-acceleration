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
    Click-to-polyline authoring (no drag):

      Left-click: add a point.
        - 1st click: starts a wire (no paint yet)
        - 2nd click: paint segment p1->p2 by dropping stamps along the line
        - 3rd click: paint p2->p3, etc.
      Right-click / Backspace: undo last segment (removes its stamps and last point)

    Hotkeys:
      a/d: rotate donor axis (−/+ 3°)
      w/s: donor separation ( +/− 2 px )
      [ / ]: block (kernel) size −/+
      h: toggle HARD PASTE vs. alpha blend (preview text only; does not affect server)
      , / .: alpha −/+  (preview text only)
      m: toggle magnifier   z/x: magnifier scale −/+
      n: next wire (start a fresh wire, keep params)
      f: next frame
      q/ESC: quit session
    """
    def __init__(self, bgr_img, frame_idx=1, total_frames=1,
                 init_angle=90.0, init_sep=14, init_block=21, alpha=0.65,
                 mag_pad=50, mag_scale=6, mag_max=480,
                 step_px=None):
        self.frame=bgr_img; self.h,self.w=bgr_img.shape[:2]
        self.cursor=(self.w//2,self.h//2)
        self.angle=float(init_angle); self.sep=float(init_sep)
        self.block=odd(int(init_block)); self.alpha=float(alpha)
        self.hard_paste=False
        self.preview_on=True

        # polyline + wires
        self.step_px = float(step_px) if step_px is not None else max(1.0, odd(int(init_block)) * 0.5)
        self.wires=[]   # each wire: dict(points=[(x,y),...], seg_stack=[(start_idx,end_idx)], stamps=[...])
        self._start_new_wire()

        # Magnifier
        self.mag_on=True; self.mag_pad=int(mag_pad)
        self.mag_scale=int(mag_scale); self.mag_max=int(mag_max)

        self.action="next_frame"   # or "quit"
        self.frame_idx=frame_idx; self.total_frames=total_frames

        self.win="Stamp Designer"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, min(self.w,1280), min(self.h,720))
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ---- wire utilities ----
    def _start_new_wire(self):
        self.wires.append(dict(points=[], seg_stack=[], stamps=[]))
        self.cur = self.wires[-1]

    def _all_stamps(self):
        # flatten all wires' stamps
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

    # ---- stamping primitives ----
    def _commit_stamp(self,x,y):
        self.cur["stamps"].append(dict(cx=float(x), cy=float(y),
                                       angle_deg=float(self.angle),
                                       sep_px=float(self.sep),
                                       bw=int(self.block), bh=int(self.block)))

    def _stamps_along_segment(self, x0,y0, x1,y1):
        # drop stamps every self.step_px along the segment (including endpoint)
        dx=x1-x0; dy=y1-y0
        L=math.hypot(dx,dy)
        if L<=1e-6: return 0
        step=max(1.0, float(self.step_px))
        n=max(1, int(math.floor(L/step)))
        # remember where this segment’s stamps start/end (for undo)
        start=len(self.cur["stamps"])
        for i in range(n+1):
            t = i / float(n)
            xs = x0 + dx * t
            ys = y0 + dy * t
            self._commit_stamp(xs, ys)
        end=len(self.cur["stamps"])
        self.cur["seg_stack"].append((start,end))
        return (end-start)

    def _add_point_and_maybe_segment(self,x,y):
        pts=self.cur["points"]
        if not pts:
            pts.append((x,y))
        else:
            x0,y0=pts[-1]
            pts.append((x,y))
            self._stamps_along_segment(x0,y0,x,y)

    def _undo_last_segment(self):
        if not self.cur["seg_stack"]:
            # if no segments in this wire, try removing last point to restart
            if self.cur["points"]:
                self.cur["points"].pop()
            return
        start,end=self.cur["seg_stack"].pop()
        # drop stamps
        del self.cur["stamps"][start:end]
        # drop last point (keeps the previous point as new tail)
        if self.cur["points"]:
            self.cur["points"].pop()

    # ---- drawing ----
    def _draw_preview(self, img):
        out=img.copy()

        # draw existing wires (rects + polylines)
        for wi,w in enumerate(self.wires):
            col = (0,200,255) if (w is self.cur) else (120,120,120)
            # rectangles for stamps (thin boxes)
            for st in w["stamps"]:
                scx,scy=int(round(st["cx"])),int(round(st["cy"]))
                bw2,bh2=int(st["bw"]),int(st["bh"])
                x0=clamp(scx-bw2//2,0,self.w-1); y0=clamp(scy-bh2//2,0,self.h-1)
                x1=clamp(scx+bw2//2+1,1,self.w); y1=clamp(scy+bh2//2+1,1,self.h)
                cv2.rectangle(out,(x0,y0),(x1-1,y1-1),col,1)

            # polyline points/segments
            pts=w["points"]
            for i,p in enumerate(pts):
                cv2.circle(out,(int(p[0]),int(p[1])),3,(0,255,0),-1)
                if i>0:
                    cv2.line(out,(int(pts[i-1][0]),int(pts[i-1][1])),(int(p[0]),int(p[1])),(0,255,0),1,cv2.LINE_AA)

        # --- live donor-axis preview (2 green dots) + kernel box at cursor ---
        cx, cy = self.cursor
        ang = math.radians(self.angle)
        vx, vy = math.cos(ang), math.sin(ang)
        half = self.sep / 2.0
        xt = cx - vx*half; yt = cy - vy*half
        xb = cx + vx*half; yb = cy + vy*half

        # draw the two donor dots
        cv2.circle(out,(int(round(xt)),int(round(yt))),5,(0,255,0),2,cv2.LINE_AA)
        cv2.circle(out,(int(round(xb)),int(round(yb))),5,(0,255,0),2,cv2.LINE_AA)

        # draw the kernel rectangle centered at the cursor
        bw=odd(self.block); bh=odd(self.block)
        x0=int(round(cx-bw//2)); y0=int(round(cy-bh//2))
        x1=clamp(x0+bw,1,self.w); y1=clamp(y0+bh,1,self.h)
        x0=clamp(x0,0,self.w-1);  y0=clamp(y0,0,self.h-1)
        cv2.rectangle(out,(x0,y0),(x1-1,y1-1),(60,255,60),1,cv2.LINE_AA)

        # preview the next segment tail -> cursor for current wire
        pts=self.cur["points"]
        if pts:
            tail=pts[-1]
            cv2.line(out,(int(tail[0]),int(tail[1])),(int(self.cursor[0]),int(self.cursor[1])),(60,255,60),1,cv2.LINE_AA)

        # HUD
        total_stamps = sum(len(w["stamps"]) for w in self.wires)
        cur_stamps = len(self.cur["stamps"])
        hud=f"[F {self.frame_idx}/{self.total_frames}] [W {len(self.wires)}] angle:{self.angle:.1f}° sep:{self.sep:.1f}px block:{self.block} " \
            f"{'HARD' if self.hard_paste or self.alpha>=0.999 else f'alpha:{int(self.alpha*100)}%'} " \
            f"wire_stamps:{cur_stamps} total:{total_stamps} step:{self.step_px:.1f}px"
        cv2.putText(out,hud,(10,max(20,self.h-10)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

        # magnifier around the kernel box
        self._add_magnifier(out,out,(x0,y0,x1,y1))
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
            elif k==ord('f'): self.action="next_frame"; break  # advance frame
            elif k==ord('n'):
                # next wire (finalize current; start a new empty wire)
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
            args.magpad,args.magscale,args.magmax,
            step_px=args.step
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
    ap=argparse.ArgumentParser(description="Interactive polyline stamp authoring client.")
    ap.add_argument("host"); ap.add_argument("port",type=int)
    ap.add_argument("--frames",type=int,default=1,help="number of frames to author")
    ap.add_argument("--angle",type=float,default=90.0)
    ap.add_argument("--sep",type=float,default=14.0)
    ap.add_argument("--block",type=int,default=21)
    ap.add_argument("--alpha",type=float,default=0.65)
    ap.add_argument("--magpad",type=int,default=50)
    ap.add_argument("--magscale",type=int,default=6)
    ap.add_argument("--magmax",type=int,default=480)
    ap.add_argument("--step",type=float,default=None,help="stamp spacing along segment in pixels (default: block/2)")
    args=ap.parse_args()

    for i in range(args.frames):
        quit_all=author_one_frame(args.host,args.port,i+1,args.frames,args)
        if quit_all: break
    print("[client] done")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"[client] ERROR: {e}", file=sys.stderr); sys.exit(1)
