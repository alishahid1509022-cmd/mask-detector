# Performance & Optimization Guide

This document explains **every** optimization applied to the detection
pipeline, and - just as importantly - the reasoning behind each one, so
future changes can be evaluated against the same tradeoffs.

## 1. Threaded camera capture (higher FPS, appropriate use of threading)

**File:** `src/mask_detector/video_stream.py`

**Problem:** `cv2.VideoCapture.read()` blocks until the camera driver hands
back a frame. Calling it directly from the main detection loop means every
iteration pays that wait *in addition to* however long YOLOv8 inference
takes - the two costs stack instead of overlapping. On a fast GPU, camera
I/O latency (not the model) becomes the bottleneck.

**Fix:** A dedicated background `threading.Thread` continuously calls
`cv2.VideoCapture.read()` in a tight loop and stores only the **single most
recent frame** behind a lock - never a growing queue. The main loop's
`read_frame()` just returns that frame; it never waits on camera I/O.
Camera reads and detection/UI work now happen concurrently instead of
sequentially, which is what actually raises effective FPS on hardware
where the model is faster than the camera's own latency.

**Why a thread (and not, say, `multiprocessing`) is the *appropriate* tool
here:** `cv2.VideoCapture.read()` is I/O-bound - the underlying C++ call
releases Python's GIL while it blocks on the camera driver. A second
Python thread can therefore make real progress during that wait without
fighting the main thread for the GIL. Multiprocessing would add IPC/frame-
serialization overhead for no benefit here, since there's no CPU-bound
Python code to parallelize in the capture path.

**Why inference itself was *not* also moved to a thread:** two reasons.
First, Streamlit requires UI calls (`st.image`, `st.metric`, ...) to happen
on the main script thread; a background inference thread would need to
hand results back to the main thread anyway, adding queues/locks for
comparatively little gain. Second, unlike the blocking camera read,
YOLOv8 inference is the pipeline's actual CPU/GPU-bound work - threading
it wouldn't shrink that cost, just relocate it. This is a case where
threading would add complexity and thread-safety risk without a matching
performance win, so it was deliberately left out of scope.

**Leak prevention built into the same change:** `release()` always signals
the thread to stop and `join()`s it (with a timeout) *before* releasing the
camera handle, and the thread is also marked `daemon=True` as a last-resort
safety net. Together these guarantee a stopped or crashed detection session
never leaves an orphaned thread (or a camera handle it's still holding
open) running in the background.

## 2. Frame-rate limiting (lower CPU usage)

**Files:** `src/mask_detector/utils/fps.py` (pre-existing `FrameRateLimiter`,
now actually wired in), `src/mask_detector/app.py`

**Problem:** Without any pacing, the detection loop runs as fast as the
CPU/GPU allow. On hardware where YOLOv8 + drawing + Streamlit updates take
less than, say, 33ms, the loop would still run 100+ times a second -
burning CPU on inference calls and UI updates nobody can actually perceive
past the target frame rate.

**Fix:** `FrameRateLimiter(target_fps=config.TARGET_FPS)` sleeps out
whatever time is left in each iteration's frame budget. This caps CPU usage
on fast hardware without capping perceived FPS below `TARGET_FPS` (default
30) - it only removes *wasted* work above that ceiling. The sleep is short
enough (well under a second at any sane target FPS) that it never
noticeably delays reacting to a Stop Camera click.

## 3. Removed the PIL round-trip when displaying frames

**Files:** `src/mask_detector/app.py`, `src/mask_detector/utils/drawing.py`

**Problem:** Every frame was converted `BGR -> RGB` via `cv2.cvtColor`, then
wrapped in a `PIL.Image`, before being handed to `st.image()`. That's a
full-frame color conversion plus an extra allocation/object, every single
frame, in the hottest part of the loop.

**Fix:** `st.image(annotated, channels="BGR", ...)` - Streamlit accepts a
raw BGR numpy array directly and handles the channel order itself,
eliminating our own conversion + PIL wrapping entirely. `bgr_to_pil()` is
still available in `utils/drawing.py` for any other caller that specifically
needs a PIL image, but the hot per-frame path no longer uses it.

## 4. Model warm-up at load time (higher *perceived* FPS on the first frames)

**File:** `src/mask_detector/detector.py`

**Problem:** A model's very first forward pass is measurably slower than
every one after it - CUDA kernel selection/compilation, memory allocation,
and (on Apple Silicon) MPS graph setup all happen lazily on first use.
Without warm-up, that one-time cost lands on the first *live* webcam frame,
showing up as a visible stutter and an artificially low initial FPS reading.

**Fix:** `MaskDetector._warmup()` runs one throwaway inference on a blank
frame immediately after loading weights, inside the same
`st.cache_resource(show_spinner="Loading YOLOv8 model...")` call that
already shows a loading spinner at startup. The one-time cost is paid
once, during an already-expected wait, instead of during live detection.
Warm-up failures are logged and swallowed - never fatal, since a failed
warmup doesn't mean real inference will fail too.

## 5. Fixed a resource-cache memory leak in `app.py`

**File:** `src/mask_detector/app.py`

**Problem:** `load_alert_manager()` and `load_screenshot_manager()` were
originally `@st.cache_resource`-decorated functions that took the sidebar's
cooldown-slider values as *parameters*. `st.cache_resource` keys its cache
by argument values and **never evicts entries by default**. Every distinct
value a user dragged a cooldown slider to would permanently retain a brand
new `AlertManager` (with its own voice backend) or `ScreenshotManager` for
the rest of the process's lifetime - a slow, silent memory leak that grows
with normal UI interaction, not just heavy load.

**Fix:** Those cooldown values were removed from the cached functions'
signatures entirely (`load_alert_manager` now only takes `engine_name`,
bounded to 2 possible values; `load_screenshot_manager` takes no arguments
at all, so there's exactly one cached instance for the app's lifetime).
Both managers gained an `update_cooldown()` method (backed by
`Cooldown.set_seconds()`) that mutates the cooldown on the *existing*
cached instance instead. `main()` calls `update_cooldown()` every rerun
with the current slider value - this mirrors the pattern already used for
`detector.confidence_threshold`, which is mutated in place rather than
being part of `load_detector()`'s cache key.

## 6. Bounded memory by construction elsewhere in the pipeline

These were already true before this round of changes but are worth
stating explicitly as part of "prevent memory leaks":

- `FPSCounter` uses a `collections.deque(maxlen=window)` - a fixed-size
  rolling window, not an ever-growing list of timestamps.
- Per-frame detections (`list[Detection]`) are local to each loop
  iteration and never accumulated; only small integer/float *counters*
  (`total_detections`, `mask_count`, `confidence_sum`, ...) persist across
  iterations in the dashboard.
- `VideoStream`'s reader thread overwrites `_latest_frame` in place rather
  than appending to a queue, so a processing loop that temporarily falls
  behind the camera's frame rate does not cause memory to grow - old,
  unconsumed frames are simply discarded, never buffered.
- The camera handle and its reader thread are always released/joined via
  `VideoStream`'s context manager (`__exit__` runs on both normal exit and
  any exception), so a crashed or interrupted session can't leak either.

## 7. Keeping the UI responsive

**File:** `src/mask_detector/app.py`

Streamlit's script runner checks for a pending rerun (e.g. a Stop Camera
click) at points where the script calls Streamlit APIs (`st.image`,
`st.metric`, `st.toast`, ...) - not asynchronously at arbitrary points in
plain Python code. The detection loop already calls several such APIs
every iteration (updating the video frame and every dashboard metric), so
Streamlit gets a chance to interrupt the loop and honor a Stop click on
every single frame. This was true before this optimization pass and was
deliberately preserved: `FrameRateLimiter.wait()`'s sleep is short (a
fraction of a second at any sane target FPS) and sits *after* those calls,
so it doesn't introduce any new, longer delay before the next such
checkpoint.

## 8. Already in place from earlier phases (kept, not re-explained here)

For completeness, these existing optimizations (documented in
[README.md](../README.md#real-time-performance) and
[architecture.md](architecture.md)) remain part of the overall performance
story and weren't changed in this pass:

- Auto-selecting the fastest available device (CUDA > Apple Silicon MPS > CPU).
- FP16 half-precision inference on CUDA.
- Configurable inference resolution (`INFERENCE_IMG_SIZE`) to trade
  accuracy for speed on slower hardware.
- Loading the model and alert/screenshot managers once via
  `st.cache_resource` rather than on every Streamlit rerun.

## Summary table

| Optimization | Addresses | File(s) |
| --- | --- | --- |
| Threaded camera capture | FPS, appropriate threading | `video_stream.py` |
| Frame-rate limiting | CPU usage | `utils/fps.py`, `app.py` |
| Skip the PIL round-trip for display | CPU usage, FPS | `app.py`, `utils/drawing.py` |
| Model warm-up at load | Perceived FPS (first frames) | `detector.py` |
| Mutate-in-place cached managers | Memory leaks | `app.py`, `utils/cooldown.py`, `alerts/alert_manager.py`, `screenshot.py` |
| Bounded counters/rolling windows | Memory leaks | `utils/fps.py`, `app.py` |
| Thread join + context manager cleanup | Memory/resource leaks | `video_stream.py` |
| Streamlit yield-point checkpoints (preserved) | UI responsiveness | `app.py` |
