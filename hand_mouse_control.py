"""
Hand-Controlled Desktop Mouse
=============================

Controls the OS mouse cursor with hand gestures captured from a webcam.

Pipeline:
    Webcam (threaded capture) -> MediaPipe HandLandmarker (Tasks API)
    -> scale-invariant gesture classification -> majority-vote stabilizer
    -> One Euro Filter (smoothing) -> pyautogui (mouse control)

On first run, this script automatically downloads the hand landmark model
bundle (hand_landmarker.task, a few MB) into the same folder as this file.
No manual download step is required.

Gestures:
    MOVE          index fingertip position (default state, any other        -> move cursor
                  finger shape not matching a gesture below)
    GRAB          thumb tip pinches index tip                               -> mouse down / drag
                  (release within DRAG_ACTIVATION_DELAY_S -> click; hold
                  longer and move -> drag; holding past GRAB_MAX_HOLD_S
                  auto-releases as a safety timeout)
    SCROLL_UP     thumbs up (thumb extended pointing up, other 4 curled)    -> scroll up (held, slow)
    SCROLL_DOWN   thumbs down (thumb extended pointing down, other 4       -> scroll down (held, slow)
                  curled)
    THREE_SWEEP   index+middle+ring extended, pinky curled, swept           -> switch desktop
                  sideways                                                     (sweep right -> RIGHT desktop,
                                                                                 sweep left  -> LEFT desktop)
    LOCK_SCREEN   middle finger extended alone, other 3 fingers curled,     -> lock the screen
                  held for LOCK_HOLD_S seconds

Controls:
    q / ESC   quit
    SPACE     pause / resume mouse control (webcam preview keeps running)

Run:
    pip install -r requirements.txt
    python hand_mouse_control.py
"""

import math
import os
import sys
import time
import platform
import threading
import urllib.request
from collections import deque, Counter

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ----------------------------------------------------------------------------
# Configuration constants (tune these to taste)
# ----------------------------------------------------------------------------

CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT = 640, 480
CAM_FPS = 30

# Model bundle: auto-downloaded next to this script on first run.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Fraction of the camera frame (from each edge) treated as "dead border".
# Lets a smaller hand-movement range cover the entire screen.
FRAME_MARGIN = 0.12

# Landmark distance ratios are normalized by hand size, so these thresholds
# work regardless of how close/far the hand is from the camera.
PINCH_RATIO = 0.35              # thumb-index distance / hand_scale, for GRAB
FINGER_EXTENDED_MARGIN = 1.10   # tip must be this much farther from wrist than pip

STABILITY_FRAMES = 5            # majority-vote window for gesture confirmation
CURSOR_DEADZONE_PX = 2          # ignore cursor moves smaller than this

# Click vs. drag timing: a pinch released before DRAG_ACTIVATION_DELAY_S
# counts as a plain click (the cursor is not moved while pinched, so tiny
# hand jitter during a quick tap can't smear into an accidental drag). Hold
# past that delay and the cursor starts tracking your hand again, turning
# it into a drag. GRAB_MAX_HOLD_S is a safety timeout that force-releases
# the mouse button if a pinch is ever held unreasonably long (e.g. a stuck
# gesture reading), so you never get stuck mid-drag.
DRAG_ACTIVATION_DELAY_S = 0.25
GRAB_MAX_HOLD_S = 6.0

SCROLL_STEP = 18                # pyautogui.scroll() units per scroll tick (lower = slower)
SCROLL_INTERVAL_S = 0.06        # minimum time between scroll ticks while held (higher = slower)
DESKTOP_SWITCH_COOLDOWN_S = 1.0 # minimum time between desktop switches

# Lock screen: the gesture must be held this long before it fires, so a
# quick/accidental middle finger doesn't lock you out immediately. A
# cooldown then prevents it from re-firing while the gesture is still held.
LOCK_HOLD_S = 1.5
LOCK_COOLDOWN_S = 3.0

# Thumb up/down detection: how far (normalized by hand size) the thumb tip
# must sit above/below the wrist, with the other four fingers curled, to
# count as a deliberate thumbs-up / thumbs-down.
THUMB_VERTICAL_RATIO = 0.5

# Three-finger sweep detection: how far (fraction of frame width) and how
# fast the tracked fingertip must travel (either direction) to count as a
# deliberate swipe.
SWEEP_MIN_DX = 0.16
SWEEP_MAX_WINDOW_S = 0.5

# One Euro Filter parameters (lower beta = smoother but laggier)
ONEEURO_MINCUTOFF = 1.0
ONEEURO_BETA = 0.012
ONEEURO_DCUTOFF = 1.0

OS_NAME = platform.system()  # 'Windows', 'Darwin' (macOS), or 'Linux'

pyautogui.FAILSAFE = True   # keep on: slam cursor to a screen corner to abort
pyautogui.PAUSE = 0.0       # we manage our own timing/cooldowns

# ----------------------------------------------------------------------------
# Hand landmark indices (same layout as MediaPipe's 21-point hand model)
# ----------------------------------------------------------------------------

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Skeleton edges for drawing the hand overlay (drawn manually below instead
# of relying on the legacy mp.solutions.drawing_utils module, which has been
# unreliable/missing in recent mediapipe releases on some platforms).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle (+ palm)
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring (+ palm)
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky (+ palm)
    (0, 17),                                 # wrist to pinky base
]


# ----------------------------------------------------------------------------
# Model download: fetch the hand landmark model bundle on first run.
# ----------------------------------------------------------------------------

def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH

    print(f"Hand landmark model not found. Downloading to:\n  {MODEL_PATH}")

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, block_num * block_size * 100 // total_size)
        sys.stdout.write(f"\r  {pct}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, _progress)
        print("\nDownload complete.")
    except Exception as e:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)  # don't leave a partial/corrupt file behind
        raise RuntimeError(
            f"Failed to download hand landmark model from {MODEL_URL}.\n"
            f"Check your internet connection, or download it manually and "
            f"place it at {MODEL_PATH}.\nOriginal error: {e}"
        )
    return MODEL_PATH


# ----------------------------------------------------------------------------
# One Euro Filter — the standard low-lag pointer-smoothing algorithm
# (Casiez, Roussel, Vogel 2012 — used widely for cursor/gesture smoothing)
# ----------------------------------------------------------------------------

class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if t is not None and self.t_prev is not None:
            dt = max(t - self.t_prev, 1e-6)
            self.freq = 1.0 / dt
        self.t_prev = t

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        edx = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * abs(edx)
        a = self._alpha(cutoff)
        ex = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev = ex, edx
        return ex


# ----------------------------------------------------------------------------
# Threaded camera capture: background thread always grabs the newest frame,
# so the main loop never processes a stale, queued-up frame (keeps latency
# and CPU load down under real-time constraints).
# ----------------------------------------------------------------------------

class ThreadedCamera:
    def __init__(self, index=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera index/permissions.")
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def lm_xy(landmarks, idx):
    p = landmarks[idx]
    return np.array([p.x, p.y], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def finger_extended(landmarks, mcp_idx, pip_idx, tip_idx, wrist, margin):
    """Scale/rotation-robust check: a finger counts as extended if its tip is
    meaningfully farther from the wrist than its own pip joint is — this works
    regardless of hand rotation or distance from the camera, unlike a plain
    'tip.y < pip.y' check."""
    tip = lm_xy(landmarks, tip_idx)
    pip = lm_xy(landmarks, pip_idx)
    return dist(wrist, tip) > dist(wrist, pip) * margin


def classify_gesture(landmarks):
    """Returns (gesture_name, cursor_tracking_point).

    Priority order matters: the most geometrically distinctive shapes
    (the pinch) are checked first, and MOVE is the fallback for anything
    that doesn't match a specific gesture — including an open palm, since
    cursor tracking no longer requires the other fingers to be curled.
    """
    wrist = lm_xy(landmarks, WRIST)
    mid_mcp = lm_xy(landmarks, MIDDLE_MCP)
    hand_scale = dist(wrist, mid_mcp) + 1e-6  # normalizes for hand size / camera distance

    index_ext = finger_extended(landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP, wrist, FINGER_EXTENDED_MARGIN)
    middle_ext = finger_extended(landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP, wrist, FINGER_EXTENDED_MARGIN)
    ring_ext = finger_extended(landmarks, RING_MCP, RING_PIP, RING_TIP, wrist, FINGER_EXTENDED_MARGIN)
    pinky_ext = finger_extended(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP, wrist, FINGER_EXTENDED_MARGIN)

    thumb_tip = lm_xy(landmarks, THUMB_TIP)
    index_tip = lm_xy(landmarks, INDEX_TIP)
    middle_tip = lm_xy(landmarks, MIDDLE_TIP)

    thumb_ext = finger_extended(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP, wrist, FINGER_EXTENDED_MARGIN)
    other_four_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext

    pinch_ratio = dist(thumb_tip, index_tip) / hand_scale

    # Thumb+index pinch -> GRAB (click / drag).
    if pinch_ratio < PINCH_RATIO:
        return "GRAB", index_tip

    # Thumbs up / thumbs down: thumb extended, other four fingers curled,
    # direction decided by whether the thumb tip sits above or below the
    # wrist (image y grows downward, so "above" means a smaller y value).
    if thumb_ext and other_four_curled:
        vertical_ratio = (wrist[1] - thumb_tip[1]) / hand_scale
        if vertical_ratio > THUMB_VERTICAL_RATIO:
            return "SCROLL_UP", thumb_tip
        if vertical_ratio < -THUMB_VERTICAL_RATIO:
            return "SCROLL_DOWN", thumb_tip

    # Middle finger extended alone, other three curled -> lock screen
    # (confirmed by a hold duration in the main loop, not fired instantly).
    if middle_ext and not index_ext and not ring_ext and not pinky_ext:
        return "LOCK_SCREEN", middle_tip

    # Three fingers extended (index+middle+ring), pinky curled -> sweep
    # gesture used for desktop switching. Track the middle fingertip since
    # it's the most stable of the three for measuring lateral travel.
    if index_ext and middle_ext and ring_ext and not pinky_ext:
        return "THREE_SWEEP", middle_tip

    # Fallback: open palm, or any other hand shape — just track the index
    # fingertip and move the cursor, regardless of what the other fingers
    # are doing.
    return "MOVE", index_tip


def draw_hand_overlay(frame, landmarks, frame_w, frame_h):
    """Manual landmark/skeleton overlay (no dependency on the legacy
    mp.solutions.drawing_utils module)."""
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def switch_desktop(direction):
    """direction: 'left' or 'right'. Hotkeys are OS-dependent; Linux varies
    by desktop environment (this targets GNOME's default binding)."""
    key = "right" if direction == "right" else "left"
    if OS_NAME == "Windows":
        pyautogui.hotkey("ctrl", "win", key)
    elif OS_NAME == "Darwin":
        pyautogui.hotkey("ctrl", key)
    else:
        pyautogui.hotkey("ctrl", "alt", key)


def lock_screen():
    """Locks the OS session. Hotkeys/commands are OS-dependent; Linux varies
    by desktop environment, so a few common lock commands are tried in
    order until one succeeds."""
    if OS_NAME == "Windows":
        # Call the Win32 API directly rather than simulating the Win+L
        # keystroke — synthetic Win+L key events are unreliable (Windows
        # often swallows them depending on focus/session state), while
        # LockWorkStation() is the actual OS call the shortcut invokes.
        import ctypes
        ctypes.windll.user32.LockWorkStation()
    elif OS_NAME == "Darwin":
        # Requires the "Lock Screen" shortcut to be enabled in
        # System Settings > Keyboard > Keyboard Shortcuts.
        pyautogui.hotkey("ctrl", "cmd", "q")
    else:
        for cmd in (
            "loginctl lock-session",
            "gnome-screensaver-command -l",
            "xdg-screensaver lock",
            "dm-tool lock",
        ):
            if os.system(cmd + " >/dev/null 2>&1") == 0:
                break


# ----------------------------------------------------------------------------
# Majority-vote stabilizer: requires a gesture to "win" over the last N
# frames before it becomes the confirmed/active gesture. Kills single-frame
# misclassification flicker without adding noticeable lag.
# ----------------------------------------------------------------------------

class GestureStabilizer:
    def __init__(self, window=5):
        self.history = deque(maxlen=window)

    def push(self, gesture):
        self.history.append(gesture)
        return Counter(self.history).most_common(1)[0][0]


# ----------------------------------------------------------------------------
# Sweep tracker: while the THREE_SWEEP gesture (index+middle+ring extended)
# is held, tracks the tracked fingertip's x position over a short rolling
# time window. Fires once the net travel within that window crosses the
# threshold in either direction, reporting which way it went, then clears
# itself so it doesn't fire again until the gesture is re-formed.
# ----------------------------------------------------------------------------

class SweepTracker:
    def __init__(self, min_dx, max_window_s):
        self.min_dx = min_dx
        self.max_window_s = max_window_s
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, x, t):
        """Returns 'right', 'left', or None."""
        self.samples.append((t, x))
        while self.samples and t - self.samples[0][0] > self.max_window_s:
            self.samples.popleft()
        if len(self.samples) < 2:
            return None
        dx = self.samples[-1][1] - self.samples[0][1]
        if dx > self.min_dx:
            self.reset()
            return "right"
        if dx < -self.min_dx:
            self.reset()
            return "left"
        return None


# ----------------------------------------------------------------------------
# Coordinate mapping: normalized landmark -> screen pixel, with the "dead
# border" margin expanded so a comfortable hand range covers the full screen.
# ----------------------------------------------------------------------------

def normalized_to_screen(x, y, screen_w, screen_h, margin=FRAME_MARGIN):
    x_eff = (x - margin) / (1 - 2 * margin)
    y_eff = (y - margin) / (1 - 2 * margin)
    x_eff = min(max(x_eff, 0.0), 1.0)
    y_eff = min(max(y_eff, 0.0), 1.0)
    return x_eff * screen_w, y_eff * screen_h


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    screen_w, screen_h = pyautogui.size()

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    hand_options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)

    cam = ThreadedCamera(CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS).start()

    stabilizer = GestureStabilizer(STABILITY_FRAMES)
    sweep_tracker = SweepTracker(SWEEP_MIN_DX, SWEEP_MAX_WINDOW_S)
    filter_x = OneEuroFilter(freq=CAM_FPS, mincutoff=ONEEURO_MINCUTOFF, beta=ONEEURO_BETA, dcutoff=ONEEURO_DCUTOFF)
    filter_y = OneEuroFilter(freq=CAM_FPS, mincutoff=ONEEURO_MINCUTOFF, beta=ONEEURO_BETA, dcutoff=ONEEURO_DCUTOFF)

    prev_gesture = "MOVE"
    last_desktop_switch_time = 0.0
    last_scroll_time = 0.0
    dragging = False
    grab_start_time = None
    lock_gesture_start_time = None
    last_lock_time = 0.0
    last_cursor = None
    paused = False

    prev_t = time.time()
    fps_smooth = 0.0

    start_time = time.time()
    last_timestamp_ms = -1

    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)  # mirror view so movement feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # HandLandmarker's VIDEO mode requires strictly increasing
            # millisecond timestamps.
            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            confirmed = "MOVE"
            now = time.time()

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand_overlay(frame, landmarks, CAM_WIDTH, CAM_HEIGHT)

                raw_gesture, track_pt = classify_gesture(landmarks)
                confirmed = stabilizer.push(raw_gesture)

                sx, sy = normalized_to_screen(track_pt[0], track_pt[1], screen_w, screen_h)
                sx = filter_x.filter(sx, t=now)
                sy = filter_y.filter(sy, t=now)

                if confirmed != "THREE_SWEEP":
                    sweep_tracker.reset()

                if not paused:
                    # Cursor tracks the fingertip continuously in MOVE. In
                    # GRAB, cursor tracking only kicks in once the pinch has
                    # been held past DRAG_ACTIVATION_DELAY_S — a quick
                    # pinch+release never moves the cursor, so it can't
                    # smear into an accidental drag.
                    held_long_enough = (
                        confirmed == "GRAB" and grab_start_time is not None
                        and (now - grab_start_time) >= DRAG_ACTIVATION_DELAY_S
                    )
                    if confirmed == "MOVE" or held_long_enough:
                        if last_cursor is None or math.hypot(sx - last_cursor[0], sy - last_cursor[1]) > CURSOR_DEADZONE_PX:
                            pyautogui.moveTo(sx, sy)
                            last_cursor = (sx, sy)

                    # GRAB: press on entering, release on leaving. Whether it
                    # ends up registering as a click or a drag depends on how
                    # long it was held (see DRAG_ACTIVATION_DELAY_S above).
                    if confirmed == "GRAB" and not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                        grab_start_time = now
                    elif confirmed != "GRAB" and dragging:
                        pyautogui.mouseUp()
                        dragging = False
                        grab_start_time = None

                    # Safety timeout: force-release if a pinch is held far
                    # longer than any deliberate drag would need, so a
                    # misread/stuck gesture can never leave the button down.
                    if dragging and grab_start_time is not None and (now - grab_start_time) > GRAB_MAX_HOLD_S:
                        pyautogui.mouseUp()
                        dragging = False
                        grab_start_time = None

                    if confirmed in ("SCROLL_UP", "SCROLL_DOWN") and (now - last_scroll_time) > SCROLL_INTERVAL_S:
                        pyautogui.scroll(SCROLL_STEP if confirmed == "SCROLL_UP" else -SCROLL_STEP)
                        last_scroll_time = now

                    if confirmed == "THREE_SWEEP":
                        sx_raw, sy_raw = track_pt[0], track_pt[1]
                        direction = sweep_tracker.update(sx_raw, now)
                        if direction is not None \
                                and (now - last_desktop_switch_time) > DESKTOP_SWITCH_COOLDOWN_S:
                            switch_desktop(direction)
                            last_desktop_switch_time = now

                    if confirmed == "LOCK_SCREEN":
                        if lock_gesture_start_time is None:
                            lock_gesture_start_time = now
                        elif (now - lock_gesture_start_time) >= LOCK_HOLD_S \
                                and (now - last_lock_time) > LOCK_COOLDOWN_S:
                            lock_screen()
                            last_lock_time = now
                            lock_gesture_start_time = None
                    else:
                        lock_gesture_start_time = None

                prev_gesture = confirmed
            else:
                # No hand visible: release any held button so we never get stuck dragging.
                if dragging and not paused:
                    pyautogui.mouseUp()
                    dragging = False
                    grab_start_time = None
                sweep_tracker.reset()
                lock_gesture_start_time = None
                prev_gesture = "MOVE"

            # --- HUD ---
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            status = "PAUSED" if paused else confirmed
            color = (0, 0, 255) if paused else (0, 200, 0)
            cv2.putText(frame, f"Gesture: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "SPACE=pause  q/ESC=quit", (10, CAM_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Hand Mouse Control", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
                if paused and dragging:
                    pyautogui.mouseUp()
                    dragging = False
                    grab_start_time = None

    finally:
        if dragging:
            pyautogui.mouseUp()
        cam.stop()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
