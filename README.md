# Hand-Controlled Desktop Mouse

Control your OS mouse cursor using hand gestures captured from a webcam — no extra hardware required.

Built with a real-time computer-vision pipeline:

```
Webcam (threaded capture) → MediaPipe Hands (landmark detection)
→ scale-invariant gesture classification → majority-vote stabilizer
→ One Euro Filter (smoothing) → pyautogui (mouse control)
```

## Features

| Gesture | Shape | Action |
|---|---|---|
| **MOVE** | Default state / any unmatched shape | Move cursor (tracks index fingertip) |
| **GRAB** | Thumb pinches index tip | Click (quick tap) or drag (hold + move) |
| **SCROLL_UP** | Thumbs up, other fingers curled | Scroll up |
| **SCROLL_DOWN** | Thumbs down, other fingers curled | Scroll down |
| **THREE_SWEEP** | Index + middle + ring extended, swept sideways | Switch virtual desktop |
| **LOCK_SCREEN** | Middle finger extended alone, held | Lock the screen |

Key controls: `SPACE` pauses/resumes mouse control, `q` / `ESC` quits.

## Why it's robust

- **Scale-invariant classification** — gesture detection is normalized against hand size, so it works regardless of how close you are to the camera.
- **Majority-vote stabilizer** — a gesture must "win" over a rolling window of frames before it's confirmed, killing single-frame misclassification flicker.
- **One Euro Filter** — a low-lag smoothing filter keeps cursor movement fluid without jitter.
- **Threaded camera capture** — a background thread always grabs the newest frame so the pipeline never processes stale, queued-up input.
- **Safety timeouts** — a stuck pinch auto-releases after a few seconds, and losing hand tracking mid-drag immediately releases the mouse button, so you can never get stuck.

## Requirements

- Python 3.9+
- A webcam
- macOS, Windows, or Linux (desktop-switch and lock-screen hotkeys are OS-specific; Linux lock support depends on your desktop environment)

## Installation

```bash
git clone https://github.com/<your-username>/hand-mouse-control.git
cd hand-mouse-control
pip install -r requirements.txt
python hand_mouse_control.py
```

## Configuration

All tunable parameters (pinch sensitivity, gesture hold times, scroll speed, filter smoothing, etc.) live as named constants near the top of `hand_mouse_control.py` — no need to dig through the pipeline logic to adjust behavior.

## Platform notes

- **Desktop switching**: uses `Ctrl+Win+←/→` on Windows, `Ctrl+←/→` on macOS, and `Ctrl+Alt+←/→` on Linux (GNOME default binding — varies by desktop environment).
- **Lock screen**: calls `LockWorkStation()` directly on Windows; requires the macOS "Lock Screen" keyboard shortcut to be enabled; tries several common lock commands in sequence on Linux.

## Disclaimer

This tool moves your mouse and can trigger OS-level actions (screen lock, desktop switching). `pyautogui.FAILSAFE` is enabled — slam the cursor into a screen corner at any time to immediately abort.

## License

MIT — see [LICENSE](LICENSE).
