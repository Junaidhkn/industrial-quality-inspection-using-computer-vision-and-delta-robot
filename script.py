import cv2
import time
import threading
import torch
import numpy as np
import math
import os
from datetime import datetime
from queue import Queue, Empty
from ultralytics import YOLO
from collections import deque
import warnings
warnings.filterwarnings("ignore")

try:
    import serial
except Exception:
    serial = None
    print("pyserial not found; serial functionality disabled.")


MODEL_PATH = "./best.engine"

MODEL_CONF_FOR_OUTPUT = 0.10
IMGSZ = 640
FRAME_WIDTH = 1920
FRAME_HEIGHT = 480
CAMERA_INDEX = 0
GRID_SPACING_MM = 20.0
PIXELS_PER_MM = 8.0
NUM_LINES_EACH_SIDE = 8
TRIGGER_LINE_RATIO = 1.0 / 8.0
DIST_THRESHOLD = 120
REMOVE_AFTER = 2.0
ARDUINO_PORT = "COM4"
ARDUINO_BAUD = 115200
CAMERA_TO_MEAN_MM = 830.0 
DELTA_Y_MIN = -180.0
DELTA_Y_MAX = 180.0
START_Z = -590.0
HOME_Z  = -435.0
TAP_Z   = -593.0
SIDE_X = 170.0  
ALIGN_SPEED_MULT = 2.0
TAP_SPEED_MULT   = 3.0
SIDE_SPEED_MULT  = 2.0
RETURN_SPEED_MULT = 2.0
MIN_VALID_SPEED_MM_S = 1.0
MAX_WAIT_MS = 30000
Y0_TOL_MM = 10.0
MAX_Y_CORRECTION_MM = 140.0
MAX_QUEUE = 60
SERIAL_LATENCY_S = 0.03
SETTLE_MARGIN_S = 0.02
STRICT_LED_X = True
LED_CONF_MIN  = 0.20
PCB_CONF_MIN  = 0.20
FAULT_CONF_MIN = 0.35
LED_CLASS_NAME = "LED"
PCB_CLASS_NAME = "LED-Board"

FAULT_CLASS_NAMES = {
    "LED-scratched",
    "TVS-damaged",
    "fuse-resistor-missing",
    "led-driver-ic-damaged",
    "mosfet-damaged",
}
SERIAL_LOG_ENABLE = True
SERIAL_LOG_DIR = "./logs"
SERIAL_PRINT_IN_CONSOLE = True

def _ensure_log_dir():
    if not SERIAL_LOG_ENABLE:
        return None
    os.makedirs(SERIAL_LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(SERIAL_LOG_DIR, f"run_{ts}.log")

LOG_PATH = _ensure_log_dir()
_log_lock = threading.Lock()

def log_line(msg: str):
    line = f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} | {msg}"
    if SERIAL_PRINT_IN_CONSOLE:
        print(line)
    if SERIAL_LOG_ENABLE and LOG_PATH:
        with _log_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
log_line(f"Device: {device}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

log_line("Loading model...")
model = YOLO(MODEL_PATH, task="segment")
try:
    if device.startswith("cuda"):
        warm = np.random.randint(0, 255, (IMGSZ, IMGSZ, 3), dtype=np.uint8)
        _ = model(warm, imgsz=IMGSZ, conf=MODEL_CONF_FOR_OUTPUT, device=device, verbose=False)
except Exception:
    pass
log_line("Model ready.")

raw_names = getattr(model, "names", None)
if raw_names is None:
    raise RuntimeError("Model has no .names; cannot map classes reliably.")

if isinstance(raw_names, dict):
    MODEL_NAMES = {int(k): str(v) for k, v in raw_names.items()}
else:
    MODEL_NAMES = {i: str(n) for i, n in enumerate(list(raw_names))}

NAME_TO_ID = {name: cid for cid, name in MODEL_NAMES.items()}
LED_CLASS_ID = NAME_TO_ID.get(LED_CLASS_NAME, None)
PCB_CLASS_ID = NAME_TO_ID.get(PCB_CLASS_NAME, None)
FAULT_CLASS_IDS = {NAME_TO_ID[n] for n in FAULT_CLASS_NAMES if n in NAME_TO_ID}

log_line(f"[MODEL] classes={MODEL_NAMES}")
log_line(f"[MAP] LED='{LED_CLASS_NAME}' -> {LED_CLASS_ID}")
log_line(f"[MAP] PCB='{PCB_CLASS_NAME}' -> {PCB_CLASS_ID}")
missing_fault = [n for n in FAULT_CLASS_NAMES if n not in NAME_TO_ID]
if missing_fault:
    log_line(f"[WARN] These fault class names were NOT found in model.names: {missing_fault}")
log_line(f"[MAP] FAULT_IDS={sorted(list(FAULT_CLASS_IDS))}")
frame_queue = Queue(maxsize=6)
result_queue = Queue(maxsize=6)
stop_signal = threading.Event()


class PCBTracker:
    def __init__(self, dist_threshold=DIST_THRESHOLD, remove_after=REMOVE_AFTER):
        self.tracks = {}
        self.dist_threshold = dist_threshold
        self.remove_after = remove_after
        self._counter = 0

    def _allocate_id(self):
        tid = f"{self._counter:04d}"
        self._counter += 1
        return tid

    def update(self, detected_pcbs):
        now = time.time()
        active_ids = []

        for det in detected_pcbs:
            best_tid = None
            best_score = 0.0

            for tid, tr in self.tracks.items():
                score = self._iou(det["box"], tr["box"])
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_score < 0.05:
                best_tid = None
                best_d = float("inf")
                for tid, tr in self.tracks.items():
                    tx, ty = tr["center"]
                    dx = det["center"][0] - tx
                    dy = det["center"][1] - ty
                    d = math.hypot(dx, dy)
                    if d < best_d and d < self.dist_threshold:
                        best_d = d
                        best_tid = tid

            if best_tid is None:
                best_tid = self._allocate_id()
                self.tracks[best_tid] = {
                    "box": det["box"],
                    "center": det["center"],
                    "prev_center": det["center"],
                    "faulty": True,
                    "has_triggered": False,
                    "last_seen": now
                }
            else:
                prev_center = self.tracks[best_tid]["center"]
                self.tracks[best_tid]["prev_center"] = prev_center
                self.tracks[best_tid]["center"] = det["center"]
                self.tracks[best_tid]["box"] = det["box"]
                self.tracks[best_tid]["faulty"] = True
                self.tracks[best_tid]["last_seen"] = now

            active_ids.append(best_tid)

        to_remove = [tid for tid, tr in list(self.tracks.items())
                     if time.time() - tr["last_seen"] > self.remove_after]
        for tid in to_remove:
            del self.tracks[tid]

        return active_ids

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        inter = interW * interH
        if inter == 0:
            return 0.0
        areaA = max(1, (boxA[2]-boxA[0])) * max(1, (boxA[3]-boxA[1]))
        areaB = max(1, (boxB[2]-boxB[0])) * max(1, (boxB[3]-boxB[1]))
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

tracker = PCBTracker()

class ArduinoComm(threading.Thread):
    def __init__(self, port=ARDUINO_PORT, baud=ARDUINO_BAUD):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.ser = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.latest_speed = 0.0
        self.latest_pos_mm = None
        self._in_lines = deque(maxlen=4000)
        self._cond = threading.Condition()

    def open(self):
        if serial is None:
            log_line("pyserial not installed. ArduinoComm disabled.")
            return
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1, write_timeout=0.2)
            time.sleep(1.0)
            self.start()
            log_line(f"Opened serial {self.port} at {self.baud}")
        except Exception as e:
            log_line(f"Failed to open serial port {self.port}: {e}")
            self.ser = None

    def close(self):
        self.stop_flag.set()
        try:
            self.send_raw("STOP")
        except Exception:
            pass
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

    def run(self):
        readbuf = b""
        while not self.stop_flag.is_set():
            try:
                if self.ser is None:
                    time.sleep(0.05)
                    continue
                data = self.ser.read(256)
                if data:
                    readbuf += data
                    while b"\n" in readbuf:
                        line, readbuf = readbuf.split(b"\n", 1)
                        line = line.strip().decode("ascii", errors="ignore")
                        if line:
                            self._handle_line(line)
                else:
                    time.sleep(0.002)
            except Exception as e:
                log_line(f"ArduinoComm read error: {e}")
                time.sleep(0.2)

    def _enqueue_in(self, tup):
        with self._cond:
            self._in_lines.append(tup)
            self._cond.notify_all()

    def _handle_line(self, line: str):
        if line.startswith("ENC "):
            try:
                parts = line.split()
                mm_s = float(parts[1])
                pos_mm = float(parts[6]) if len(parts) >= 7 else None
                with self.lock:
                    if math.isfinite(mm_s):
                        self.latest_speed = mm_s
                    if pos_mm is not None and math.isfinite(pos_mm):
                        self.latest_pos_mm = pos_mm
            except Exception:
                pass
            return

        if line.startswith("STARTED "):
            _id = line[len("STARTED "):].strip()
            self._enqueue_in(("STARTED", _id))
            return
        if line.startswith("DONE "):
            _id = line[len("DONE "):].strip()
            self._enqueue_in(("DONE", _id))
            return
        if line.startswith("ACK_"):
            tok = line.split()[0].strip()
            self._enqueue_in(("ACK", tok))
            return
        self._enqueue_in(("LOG", line))
        
    def send_move_angles(self, id_str, a0, a1, a2, speedMult=1.0):
        cmd = f"MOVE_ANGLES {id_str} {a0:.3f} {a1:.3f} {a2:.3f} {speedMult:.3f}\n"
        if self.ser is None:
            log_line(f"[SERIAL DISABLED] would send: {cmd.strip()}")
            return
        with self.lock:
            try:
                self.ser.write(cmd.encode("ascii"))
            except Exception as e:
                log_line(f"Serial write error: {e}")

    def send_raw(self, raw_cmd: str):
        if self.ser is None:
            log_line(f"[SERIAL DISABLED] would send: {raw_cmd.strip()}")
            return
        with self.lock:
            try:
                out = raw_cmd if raw_cmd.endswith("\n") else raw_cmd + "\n"
                self.ser.write(out.encode("ascii"))
            except Exception as e:
                log_line(f"Serial write error (raw): {e}")

    def wait_for(self, typ, payload, timeout_ms=15000, cancel_event=None):
        deadline = time.time() + timeout_ms / 1000.0
        with self._cond:
            while time.time() < deadline:
                if self.stop_flag.is_set() or (cancel_event is not None and cancel_event.is_set()):
                    return False

                n = len(self._in_lines)
                if n:
                    for i in range(n):
                        t, p = self._in_lines[i]
                        ok = False
                        if t == typ:
                            if typ in ("STARTED", "DONE", "ACK"):
                                ok = (p == payload)
                            elif typ == "LOG":
                                ok = (payload in p)
                        if ok:
                            self._in_lines.rotate(-i)
                            _ = self._in_lines.popleft()
                            self._in_lines.rotate(i)
                            return True

                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=min(0.2, remaining))
            return False

    def get_latest_speed(self) -> float:
        with self.lock:
            s = float(self.latest_speed)
        if not math.isfinite(s):
            return 0.0
        return s

    def get_latest_pos_mm(self):
        with self.lock:
            return self.latest_pos_mm

class DeltaIK:
    def __init__(self):
        self.f = 135.0
        self.e = 35.0 
        self.rf = 250.0
        self.re = 515.0
        self.SQRT3 = 1.732050807
        self.T = (2.0 * self.f - self.e) * self.SQRT3 / 3.0

    def calcAngleYZ(self, x0, y0, z0):
        if abs(z0) < 1e-12:
            raise ValueError("z0 too small")
        yp = y0 + self.T
        A = (x0*x0 + yp*yp + z0*z0 + self.rf*self.rf - self.re*self.re) / (2.0 * z0)
        B = yp / z0
        D = self.rf*self.rf - (A - B*yp)*(A - B*yp) / (1.0 + B*B)
        if D < 0:
            raise ValueError("D < 0 (no solution)")
        yj = (A*B + yp - math.sqrt(D*(1.0 + B*B))) / (1.0 + B*B)
        zj = A - B*yj
        theta = -math.degrees(math.atan2(-zj, -yj))
        if not math.isfinite(theta):
            raise ValueError("theta not finite")
        return theta

    def inverse(self, x, y, z):
        t1 = self.calcAngleYZ(x, y, z)
        x2 = x * math.cos(2*math.pi/3) + y * math.sin(2*math.pi/3)
        y2 = -x * math.sin(2*math.pi/3) + y * math.cos(2*math.pi/3)
        t2 = self.calcAngleYZ(x2, y2, z)
        x3 = x * math.cos(4*math.pi/3) + y * math.sin(4*math.pi/3)
        y3 = -x * math.sin(4*math.pi/3) + y * math.cos(4*math.pi/3)
        t3 = self.calcAngleYZ(x3, y3, z)
        return (t1, t2, t3)

class FaultQueue:
    def __init__(self):
        self.lock = threading.Lock()
        self.q = deque()

    def enqueue(self, item):
        with self.lock:
            if len(self.q) >= MAX_QUEUE:
                _ = self.q.popleft()
            self.q.append(item)

    def peek(self):
        with self.lock:
            return self.q[0] if self.q else None

    def dequeue(self):
        with self.lock:
            return self.q.popleft() if self.q else None

    def is_empty(self):
        with self.lock:
            return len(self.q) == 0

class Scheduler(threading.Thread):
    def __init__(self, arduino_comm: ArduinoComm, ik: DeltaIK, queue: FaultQueue):
        super().__init__(daemon=True)
        self.ard = arduino_comm
        self.ik = ik
        self.queue = queue
        self.stop_flag = threading.Event()
        self._ema_align_s = 0.40
        self._ema_tap_s   = 0.25
        self._ema_side_s  = 0.50
        self._ema_return_s = 0.45
        self._ema_alpha   = 0.25

    def stop(self):
        self.stop_flag.set()

    def _ema_update(self, attr, new_val):
        old = float(getattr(self, attr))
        a = self._ema_alpha
        val = (1.0 - a) * old + a * float(new_val)
        setattr(self, attr, val)

    def _get_speed(self) -> float:
        s = self.ard.get_latest_speed() if self.ard else 0.0
        if not math.isfinite(s) or s < 0:
            return 0.0
        return s

    def _get_pos(self):
        return self.ard.get_latest_pos_mm() if self.ard else None

    def _clamp_y(self, y):
        y = max(min(y, DELTA_Y_MAX), DELTA_Y_MIN)
        y = max(min(y, MAX_Y_CORRECTION_MM), -MAX_Y_CORRECTION_MM)
        return y

    def _item_y_from_encoder(self, pos_now, pos_det):
        return (pos_now - pos_det) - CAMERA_TO_MEAN_MM

    def _item_y_from_time(self, t_now, t_detect, speed_mm_s):
        return (speed_mm_s * (t_now - t_detect)) - CAMERA_TO_MEAN_MM

    def _do_move(self, move_id, angles, speed_mult, timeout_ms):
        if self.stop_flag.is_set():
            return False, 0.0
        t0 = time.time()
        self.ard.send_move_angles(move_id, angles[0], angles[1], angles[2], speed_mult)
        _ = self.ard.wait_for("STARTED", move_id, timeout_ms=2500, cancel_event=self.stop_flag)
        ok = self.ard.wait_for("DONE", move_id, timeout_ms=timeout_ms, cancel_event=self.stop_flag)
        dur = max(0.0, time.time() - t0)
        if not ok and not self.stop_flag.is_set():
            log_line(f"[SCHED] TIMEOUT DONE {move_id} -> sending STOP")
            try:
                self.ard.send_raw("STOP")
            except Exception:
                pass
        return ok, dur

    def _wait_until(self, t_target, max_ms=MAX_WAIT_MS):
        deadline = min(t_target, time.time() + max_ms / 1000.0)
        while not self.stop_flag.is_set():
            now = time.time()
            if now >= deadline:
                return True
            time.sleep(0.005)
        return False

    def run(self):
        while not self.stop_flag.is_set():
            if self.queue.is_empty():
                time.sleep(0.01)
                continue

            item = self.queue.peek()
            if item is None:
                time.sleep(0.01)
                continue

            base_id = str(item["id"])
            x = float(item["x_mm"])
            t_detect = float(item["t_detect"])
            pos_det = item.get("pos_det_mm", None)
            speed_det = float(item.get("speed_mm_s", 0.0))

            v = speed_det if speed_det >= MIN_VALID_SPEED_MM_S else self._get_speed()
            if v < MIN_VALID_SPEED_MM_S:
                v = max(10.0, v)

            t_pick_ideal = t_detect + (CAMERA_TO_MEAN_MM / v)
            lead_align = self._ema_align_s + SERIAL_LATENCY_S + SETTLE_MARGIN_S
            t_start_align = t_pick_ideal - lead_align
            now = time.time()
            if t_start_align > now:
                while not self.stop_flag.is_set():
                    now2 = time.time()
                    if now2 >= t_start_align:
                        break
                    pos_now = self._get_pos()
                    if pos_now is not None and pos_det is not None:
                        y_now = self._item_y_from_encoder(pos_now, pos_det)
                        if abs(y_now) <= (Y0_TOL_MM * 2.0):
                            break
                    time.sleep(0.01)

            if self.stop_flag.is_set():
                break
            try:
                ang_align = self.ik.inverse(x, 0.0, HOME_Z)
            except Exception as e:
                log_line(f"[SCHED] IK ALIGN failed id={base_id} x={x:.1f}: {e} -> drop item")
                _ = self.queue.dequeue()
                continue

            log_line(f"[SCHED] ALIGN_X id={base_id} -> [x={x:+.1f}, y=0, z=HOME_Z] t_pick_ideal={t_pick_ideal-now:+.2f}s")
            ok, dur_align = self._do_move(f"{base_id}_AX", ang_align, ALIGN_SPEED_MULT, timeout_ms=20000)
            self._ema_update("_ema_align_s", dur_align)
            if not ok:
                _ = self.queue.dequeue()
                continue
            pos_now2 = self._get_pos()
            now3 = time.time()

            if pos_now2 is not None and pos_det is not None:
                y_now = self._item_y_from_encoder(pos_now2, pos_det)
            else:
                y_now = self._item_y_from_time(now3, t_detect, v)
            if y_now > (DELTA_Y_MAX + 30.0):
                log_line(f"[SCHED] MISS id={base_id} y_now={y_now:.1f} (too late) -> drop")
                _ = self.queue.dequeue()
                continue
            if y_now < -Y0_TOL_MM:
                wait_start = time.time()
                while not self.stop_flag.is_set():
                    pos_now_w = self._get_pos()
                    t_now_w = time.time()
                    if pos_now_w is not None and pos_det is not None:
                        y_w = self._item_y_from_encoder(pos_now_w, pos_det)
                    else:
                        y_w = self._item_y_from_time(t_now_w, t_detect, v)

                    if y_w >= -Y0_TOL_MM:
                        y_now = y_w
                        break

                    if (time.time() - wait_start) * 1000.0 > MAX_WAIT_MS:
                        break
                    time.sleep(0.005)
            v_now = self._get_speed()
            if v_now >= MIN_VALID_SPEED_MM_S:
                v_used = v_now
            else:
                v_used = v

            lead_tap = SERIAL_LATENCY_S + 0.5 * max(0.05, self._ema_tap_s)  # half segment as "start"
            y_for_tap = self._clamp_y(y_now + v_used * lead_tap)
            try:
                ang_down = self.ik.inverse(x, y_for_tap, TAP_Z)
                ang_up   = self.ik.inverse(x, y_for_tap, HOME_Z)
            except Exception as e:
                log_line(f"[SCHED] IK TAP failed id={base_id} x={x:.1f} y={y_for_tap:.1f}: {e} -> drop")
                _ = self.queue.dequeue()
                continue
            t_item_start = time.time()
            log_line(f"[SCHED] TAP_DOWN id={base_id} -> y={y_for_tap:+.1f}")
            ok_d, dur_d = self._do_move(f"{base_id}_D", ang_down, TAP_SPEED_MULT, timeout_ms=15000)
            if not ok_d:
                _ = self.queue.dequeue()
                continue

            log_line(f"[SCHED] TAP_UP id={base_id}")
            ok_u, dur_u = self._do_move(f"{base_id}_U", ang_up, TAP_SPEED_MULT, timeout_ms=15000)
            dur_tap = dur_d + dur_u
            self._ema_update("_ema_tap_s", dur_tap)
            if not ok_u:
                _ = self.queue.dequeue()
                continue
            x_drop = SIDE_X if x >= 0 else -SIDE_X
            try:
                ang_side = self.ik.inverse(x_drop, 0.0, HOME_Z)
            except Exception as e:
                log_line(f"[SCHED] IK SIDE failed id={base_id}: {e}")
                ang_side = None

            dur_side = 0.0
            if ang_side is not None:
                log_line(f"[SCHED] SIDE_DROP id={base_id} -> x_drop={x_drop:+.1f}")
                ok_s, dur_side = self._do_move(f"{base_id}_S", ang_side, SIDE_SPEED_MULT, timeout_ms=20000)
                self._ema_update("_ema_side_s", dur_side)
                if ok_s:
                    try:
                        self.ard.send_raw("SOL_OFF @py")
                    except Exception:
                        pass
            try:
                ang_home = self.ik.inverse(0.0, 0.0, HOME_Z)
                log_line(f"[SCHED] RETURN_HOME id={base_id}")
                ok_h, dur_home = self._do_move(f"{base_id}_H", ang_home, RETURN_SPEED_MULT, timeout_ms=20000)
                self._ema_update("_ema_return_s", dur_home)
                if ok_h:
                    try:
                        self.ard.send_raw("SOL_ON @py")
                    except Exception:
                        pass
            except Exception as e:
                log_line(f"[SCHED] IK HOME failed id={base_id}: {e}")
                dur_home = 0.0

            total = max(0.0, time.time() - t_item_start)
            log_line(
                f"[TIMING] id={base_id} "
                f"align={dur_align:.3f}s tap={dur_tap:.3f}s side={dur_side:.3f}s return={dur_home:.3f}s total={total:.3f}s "
                f"(ema_align={self._ema_align_s:.3f}, ema_tap={self._ema_tap_s:.3f}, ema_side={self._ema_side_s:.3f}, ema_return={self._ema_return_s:.3f})"
            )

            _ = self.queue.dequeue()
            time.sleep(0.005)
def start_system(serial_port=ARDUINO_PORT, fq: FaultQueue = None):
    if fq is None:
        fq = FaultQueue()

    ard = ArduinoComm(port=serial_port)
    ard.open()
    ik = DeltaIK()

    time.sleep(0.4)
    try:
        ard.send_raw("TELEM_ON @py")
        _ = ard.wait_for("ACK", "ACK_TELEM_ON", timeout_ms=1500, cancel_event=stop_signal)
    except Exception as e:
        log_line(f"[STARTUP] TELEM_ON failed: {e}")
    try:
        ard.send_raw("SOL_OFF @py")
        _ = ard.wait_for("ACK", "ACK_SOL_OFF", timeout_ms=1500, cancel_event=stop_signal)
    except Exception:
        pass
    try:
        startup_angles = ik.inverse(0.0, 0.0, START_Z)
        ard.send_raw(f"SET_POS {startup_angles[0]:.3f} {startup_angles[1]:.3f} {startup_angles[2]:.3f} @py")
        _ = ard.wait_for("ACK", "ACK_SET_POS", timeout_ms=2500, cancel_event=stop_signal)
    except Exception as e:
        log_line(f"[STARTUP] SET_POS failed: {e}")

    try:
        home_angles = ik.inverse(0.0, 0.0, HOME_Z)
        move_id = "INIT_H"
        ard.send_move_angles(move_id, home_angles[0], home_angles[1], home_angles[2], 1.0)
        ard.wait_for("STARTED", move_id, timeout_ms=3000, cancel_event=stop_signal)
        done = ard.wait_for("DONE", move_id, timeout_ms=30000, cancel_event=stop_signal)
        log_line(f"[STARTUP] INIT_H done={done}")
        if done:
            try:
                ard.send_raw("SOL_ON @py")
                _ = ard.wait_for("ACK", "ACK_SOL_ON", timeout_ms=1500, cancel_event=stop_signal)
            except Exception:
                pass
    except Exception as e:
        log_line(f"[STARTUP] INIT_H failed: {e}")

    sched = Scheduler(ard, ik, fq)
    sched.start()
    log_line("[STARTUP] Scheduler started.")
    return ard, ik, fq, sched
def process_model_results(results, scale_x, scale_y):
    if not results or len(results) == 0:
        return [], [], []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return [], [], []

    try:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy().astype(float)
    except Exception as e:
        log_line(f"Model output parsing error: {e}")
        return [], [], []

    leds, faults, pcbs = [], [], []

    for cid, conf, b in zip(cls_ids, confs, xyxy):
        conf = float(conf)
        name = MODEL_NAMES.get(int(cid), f"Class{int(cid)}")

        x1 = int(round(b[0] * scale_x)); y1 = int(round(b[1] * scale_y))
        x2 = int(round(b[2] * scale_x)); y2 = int(round(b[3] * scale_y))
        box = [x1, y1, x2, y2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if name == LED_CLASS_NAME and conf >= LED_CONF_MIN:
            leds.append({"box": box, "center": (cx, cy), "conf": conf})
        elif name == PCB_CLASS_NAME and conf >= PCB_CONF_MIN:
            pcbs.append({"box": box, "center": (cx, cy), "led_center": None, "is_faulty": False})
        elif (int(cid) in FAULT_CLASS_IDS) and conf >= FAULT_CONF_MIN:
            faults.append({"box": box, "center": (cx, cy), "class_id": int(cid), "class_name": name, "conf": conf})
    for pcb in pcbs:
        bx1, by1, bx2, by2 = pcb["box"]
        pcx, pcy = pcb["center"]
        best = None
        best_d = float("inf")
        for led in leds:
            lcx, lcy = led["center"]
            if bx1 <= lcx <= bx2 and by1 <= lcy <= by2:
                d = math.hypot(lcx - pcx, lcy - pcy)
                if d < best_d:
                    best_d = d
                    best = (lcx, lcy)
        pcb["led_center"] = best

    for pcb in pcbs:
        bx1, by1, bx2, by2 = pcb["box"]
        pcb["is_faulty"] = False
        for f in faults:
            fx, fy = f["center"]
            if bx1 <= fx <= bx2 and by1 <= fy <= by2:
                pcb["is_faulty"] = True
                break

    return leds, faults, pcbs
def capture_thread(cap):
    while not stop_signal.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put(frame)
    cap.release()

def inference_thread():
    while not stop_signal.is_set():
        try:
            frame = frame_queue.get(timeout=0.2)
        except Empty:
            continue
        small = cv2.resize(frame, (IMGSZ, IMGSZ))
        try:
            results = model(small, imgsz=IMGSZ, conf=MODEL_CONF_FOR_OUTPUT, device=device, verbose=False, retina_masks=True)
        except Exception:
            continue

        scale_x = FRAME_WIDTH / IMGSZ
        scale_y = FRAME_HEIGHT / IMGSZ
        leds, faults, pcbs = process_model_results(results, scale_x, scale_y)

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Empty:
                pass
        result_queue.put((frame, leds, faults, pcbs))
        
def draw_grid_and_trigger(frame):
    h, w = frame.shape[:2]
    center_x = w // 2
    pixels_between = max(1, int(round(GRID_SPACING_MM * PIXELS_PER_MM)))

    for offset in range(-NUM_LINES_EACH_SIDE, NUM_LINES_EACH_SIDE + 1):
        x = center_x + offset * pixels_between
        if 0 <= x < w:
            if offset == 0:
                cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 2)
            else:
                cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)

    trigger_y = int(h * TRIGGER_LINE_RATIO)
    cv2.line(frame, (0, trigger_y), (w, trigger_y), (255, 0, 0), 2)
    return center_x, trigger_y

def enqueue_fault(fq: FaultQueue, ard: ArduinoComm, track_id: str, x_mm: float):
    tnow = time.time()
    pos_snapshot = ard.get_latest_pos_mm() if ard else None
    speed_snapshot = ard.get_latest_speed() if ard else 0.0

    fq.enqueue({
        "id": track_id,
        "x_mm": float(x_mm),
        "t_detect": tnow,
        "pos_det_mm": pos_snapshot,
        "speed_mm_s": float(speed_snapshot)
    })
    log_line(f"[VISION] ENQUEUE id={track_id} x={x_mm:+.2f} pos_det={pos_snapshot} speed={speed_snapshot:.2f}")

def process_and_draw(frame, leds, faults, pcbs, center_x, trigger_y, fq: FaultQueue, ard: ArduinoComm):
    for led in leds:
        x1, y1, x2, y2 = led["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, "LED", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for f in faults:
        x1, y1, x2, y2 = f["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f["class_name"], (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for pcb in pcbs:
        if pcb["is_faulty"]:
            x1, y1, x2, y2 = pcb["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    detected_faulty_pcbs = []
    for pcb in pcbs:
        if not pcb["is_faulty"]:
            continue
        led_center = pcb.get("led_center", None)
        if STRICT_LED_X and led_center is None:
            continue
        center = led_center if led_center is not None else pcb["center"]
        detected_faulty_pcbs.append({"box": pcb["box"], "center": center, "is_faulty": True})

    active_ids = tracker.update(detected_faulty_pcbs)

    for tid in active_ids:
        tr = tracker.tracks.get(tid)
        if tr is None:
            continue

        x1, y1, x2, y2 = tr["box"]
        cx, cy = tr["center"]
        px, py = tr.get("prev_center", (cx, cy))

        crossed = (py < trigger_y <= cy) or (py > trigger_y >= cy)
        bbox_cuts = (y1 < trigger_y < y2)

        if tr.get("faulty", True) and (not tr.get("has_triggered", False)) and (crossed or bbox_cuts):
            pixel_dx = cx - center_x
            mm_dx = pixel_dx / PIXELS_PER_MM
            log_line(f"[VISION] TRIGGER id={tid} LED_x pixel_dx={pixel_dx:+.1f} -> x_mm={mm_dx:+.2f}")
            enqueue_fault(fq, ard, tid, mm_dx)
            tr["has_triggered"] = True

        cv2.putText(frame, f"ID:{tid}",
                    (max(1, x1), min(FRAME_HEIGHT - 5, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return frame

def main():
    log_line("========== RUN START ==========")
    if LOG_PATH:
        log_line(f"Log file: {LOG_PATH}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        log_line("ERROR: Could not open camera at index 0.")
        return

    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        log_line("ERROR: Camera opened but did not return frames.")
        cap.release()
        return

    fq = FaultQueue()
    ard = None
    ik = None
    sched = None

    tcap = threading.Thread(target=capture_thread, args=(cap,), daemon=True)
    tinf = threading.Thread(target=inference_thread, daemon=True)
    tcap.start()
    time.sleep(0.1)
    tinf.start()

    try:
        ard, ik, _, sched = start_system(serial_port=ARDUINO_PORT, fq=fq)
    except Exception as e:
        log_line(f"Failed to start system: {e}")
        ard = None
        ik = None

    cv2.namedWindow("PCB Inspector", cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                frame, leds, faults, pcbs = result_queue.get(timeout=0.2)
            except Empty:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            center_x, trigger_y = draw_grid_and_trigger(frame)
            frame = process_and_draw(frame, leds, faults, pcbs, center_x, trigger_y, fq, ard)
            cv2.imshow("PCB Inspector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if ard and ik:
                    try:
                        log_line("[MAIN] q pressed: moving robot to START_Z before exit.")
                        try:
                            ard.send_raw("SOL_OFF @py")
                        except Exception:
                            pass
                        start_angles = ik.inverse(0.0, 0.0, START_Z)
                        move_id = "QUIT_TO_START"
                        ard.send_move_angles(move_id, start_angles[0], start_angles[1], start_angles[2], 1.0)
                        ard.wait_for("STARTED", move_id, timeout_ms=3000)
                        _ = ard.wait_for("DONE", move_id, timeout_ms=30000)
                    except Exception as e:
                        log_line(f"[MAIN] failed to move to START_Z on quit: {e}")
                break

    finally:
        stop_signal.set()
        tcap.join(timeout=1.0)
        tinf.join(timeout=1.0)

        try:
            if sched:
                sched.stop()
        except Exception:
            pass

        try:
            if ard:
                try:
                    ard.send_raw("TELEM_OFF @py")
                    _ = ard.wait_for("ACK", "ACK_TELEM_OFF", timeout_ms=1000)
                except Exception:
                    pass
                ard.close()
        except Exception:
            pass

        time.sleep(0.1)
        cv2.destroyAllWindows()
        log_line("========== RUN STOP ==========")

if __name__ == "__main__":
    main()
