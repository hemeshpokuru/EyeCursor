
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import math

# ==========================================================
# EYE-CONTROLLED MOUSE - CORRECTED VERSION
# Windows | Python 3.11 | OpenCV | MediaPipe | PyAutoGUI
# ==========================================================

CAMERA_INDEX = 0

# Cursor
SMOOTHING = 0.20

# Dwell click
DWELL_TIME = 1.4
DWELL_RADIUS = 24
DWELL_REARM_DISTANCE = 65

# Blink click
BLINK_THRESHOLD = 0.18
MIN_BLINK_TIME = 0.10
MAX_BLINK_TIME = 0.65

# Calibration
CALIBRATION_HOLD = 2.5
CALIBRATION_WARMUP = 0.35
MIN_CALIBRATION_SAMPLES = 5
# A deliberately generous lower bound.  It rejects closed / nearly closed
# eyes, but does not use the blink threshold as a calibration requirement.
CALIBRATION_MIN_EAR = 0.10
# Robust calibration quality limits, expressed in normalized iris position.
# The central 80% of a steady target should not cover more than 6% of the eye
# opening.  MAD is resistant to individual landmark spikes.
CALIBRATION_OUTLIER_Z = 3.5
CALIBRATION_NOISE_FLOOR = 0.003
MAX_CALIBRATION_CENTRAL_SPREAD = 0.060
MAX_CALIBRATION_MAD = 0.025
MIN_CALIBRATION_INLIER_RATIO = 0.65
MAX_TARGET_ATTEMPTS = 3
# Pairwise target separation is measured after scaling each gaze axis by the
# observed across-target standard deviation (see calibration_targets_are_distinct).
MIN_TARGET_GAZE_DISTANCE = 0.08
MIN_LID_OPEN_RATIO = 0.08
MAX_CALIBRATION_CONDITION = 10000

# Preview
PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540
WINDOW = "Eye Cursor Control"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = 468
RIGHT_IRIS = 473

EYE_CORNERS = [
    (33, 133, LEFT_IRIS),
    (362, 263, RIGHT_IRIS),
]

EYE_LIDS = [
    (159, 145, LEFT_IRIS),
    (386, 374, RIGHT_IRIS),
]

EAR_POINTS = [
    (33, 160, 158, 133, 153, 144),
    (362, 385, 387, 263, 373, 380),
]


# ==========================================================
# 1. EYE MEASUREMENTS
# ==========================================================

def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def get_eye_aspect_ratio(points, indices):
    """
    Calculate EAR from PIXEL coordinates.
    This avoids distortion caused by normalized coordinates.
    """

    p1, p2, p3, p4, p5, p6 = [
        points[i] for i in indices
    ]

    horizontal = distance(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


def get_gaze_features(face_landmarks):
    """
    Calculate iris motion relative to the eye-corner midpoint and eye width.

    Eyelid separation changes during blinking and varies even during a steady
    gaze, so using it as the vertical denominator magnifies landmark jitter.
    We only use the lids to reject a nearly closed eye; both gaze axes use the
    much more stable corner-to-corner scale.
    """

    if len(face_landmarks.landmark) <= RIGHT_IRIS:
        # This means Face Mesh was run without refined landmarks.  Iris IDs
        # 468--477 are not present in the standard 468-point mesh.
        return None, {"reason": "Refined iris landmarks unavailable"}

    points = [(lm.x, lm.y) for lm in face_landmarks.landmark]

    eye_features = []
    lid_ratios = []
    for (corner_a, corner_b, iris_index), (upper, lower, _) in zip(
            EYE_CORNERS, EYE_LIDS):
        ax, ay = points[corner_a]
        bx, by = points[corner_b]
        # Use the corner axis and its perpendicular as an eye-local frame.
        # This compensates for face roll and keeps both eyes in the same
        # anatomical orientation despite the mirrored image coordinates.
        dx, dy = bx - ax, by - ay
        eye_width = math.hypot(dx, dy)
        lid_ratio = abs(points[lower][1] - points[upper][1]) / max(eye_width, 1e-6)
        lid_ratios.append(lid_ratio)
        if eye_width < 0.01 or lid_ratio < MIN_LID_OPEN_RATIO:
            continue
        ux, uy = dx / eye_width, dy / eye_width
        if ux < 0.0:
            ux, uy = -ux, -uy
        # Landmark 468/473 is the iris-center point when refined landmarks
        # are enabled. Project it onto eye width and the perpendicular axis.
        ix, iy = points[iris_index]
        rx, ry = ix - (ax + bx) / 2.0, iy - (ay + by) / 2.0
        horizontal = (rx * ux + ry * uy) / eye_width
        vertical = (rx * -uy + ry * ux) / eye_width
        eye_features.append((horizontal, vertical))

    if len(eye_features) != 2:
        return None, {"reason": "Eye closed or unreliable",
                      "lid_ratios": lid_ratios, "eyes": len(eye_features)}

    # Average both eye-local measurements. Median is robust to one eye's
    # landmark jitter while preserving the shared gaze direction.
    gaze = np.median(np.asarray(eye_features, dtype=np.float64), axis=0)
    quality = {"reason": "Tracking", "lid_ratios": lid_ratios, "eyes": 2,
               "eye_features": eye_features}
    return (gaze, quality) if np.all(np.isfinite(gaze)) else (
        None, {"reason": "Non-finite gaze", "lid_ratios": lid_ratios, "eyes": 0})


def get_eye_measurements(face_landmarks, frame_width, frame_height):
    landmarks = face_landmarks.landmark

    if len(landmarks) <= RIGHT_IRIS:
        return None, None, {"reason": "Refined iris landmarks unavailable"}

    if not all(math.isfinite(v) for lm in landmarks
               for v in (lm.x, lm.y, lm.z)):
        return None, None, {"reason": "Non-finite landmarks"}

    # IMPORTANT: Convert normalized points to pixel coordinates
    # before calculating eye aspect ratio.
    pixel_points = [
        (
            lm.x * frame_width,
            lm.y * frame_height
        )
        for lm in landmarks
    ]

    ear_values = [
        get_eye_aspect_ratio(pixel_points, indices)
        for indices in EAR_POINTS
    ]

    ear = float(np.mean(ear_values))
    gaze, quality = get_gaze_features(face_landmarks)

    return gaze, ear if math.isfinite(ear) else None, quality


# ==========================================================
# 2. CAMERA
# ==========================================================

def open_camera():
    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open webcam. Close other camera apps "
            "or change CAMERA_INDEX to 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap


def read_face(cap, face_mesh):
    success, frame = cap.read()

    if not success:
        return None, None, None, None, "Camera frame unavailable", {}

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return frame, None, None, None, "Face not detected", {}

    face = result.multi_face_landmarks[0]

    gaze, ear, quality = get_eye_measurements(
        face, width, height
    )

    if gaze is None:
        reason = quality["reason"]
    elif ear is None:
        reason = "Invalid eye-openness measurement"
    else:
        reason = "Tracking"

    return frame, gaze, ear, face, reason, quality


# ==========================================================
# 3. PREVIEW
# ==========================================================

def draw_tracking_landmarks(frame, face):
    """Draw the landmarks used by the gaze calculation on a frame copy."""
    display_frame = frame.copy()
    if face is not None:
        h, w = display_frame.shape[:2]
        for index, colour in ((LEFT_IRIS, (0, 255, 0)),
                              (RIGHT_IRIS, (0, 255, 0)),
                              (33, (255, 200, 0)), (133, (255, 200, 0)),
                              (362, (255, 200, 0)), (263, (255, 200, 0)),
                              (159, (255, 0, 255)), (145, (255, 0, 255)),
                              (386, (255, 0, 255)), (374, (255, 0, 255))):
            if index < len(face.landmark):
                lm = face.landmark[index]
                cv2.circle(display_frame, (int(lm.x * w), int(lm.y * h)), 3, colour, -1)
    return display_frame


def show_preview(frame, gaze=None, ear=None, status="",
                 dwell_progress=0.0, sample_count=None, face=None):

    # Draw on the camera frame before resizing so landmark positions remain
    # aligned with the video rather than with the preview window.
    frame = draw_tracking_landmarks(frame, face)

    display = cv2.resize(
        frame,
        (PREVIEW_WIDTH, PREVIEW_HEIGHT)
    )

    if gaze is None:
        cv2.putText(
            display,
            "Face / iris not detected",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
    else:
        cv2.putText(
            display,
            f"Iris X: {gaze[0]:.3f}  Y: {gaze[1]:.3f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            2,
        )

    if ear is not None:
        cv2.putText(
            display,
            f"Eye openness: {ear:.3f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 220, 0),
            2,
        )

    if sample_count is not None:
        cv2.putText(
            display,
            f"Valid samples: {sample_count}",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    # Dwell progress bar
    bx, by, bw, bh = 20, 120, 220, 14

    cv2.rectangle(
        display,
        (bx, by),
        (bx + bw, by + bh),
        (180, 180, 180),
        2,
    )

    fill = int(
        bw * np.clip(dwell_progress, 0.0, 1.0)
    )

    if fill:
        cv2.rectangle(
            display,
            (bx, by),
            (bx + fill, by + bh),
            (0, 255, 0),
            -1,
        )

    cv2.putText(
        display,
        "Q/ESC: Quit | C: Calibrate | SPACE: Pause",
        (20, PREVIEW_HEIGHT - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )

    if status:
        cv2.putText(
            display,
            status[:75],
            (20, PREVIEW_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
        )

    cv2.imshow(WINDOW, display)


# ==========================================================
# 4. CALIBRATION
# ==========================================================

def polynomial_features(gaze):
    x, y = gaze

    return np.array(
        [1.0, x, y, x*x, y*y, x*y],
        dtype=np.float64,
    )


def fit_calibration(gaze_samples, screen_targets):
    # Centering and scaling makes the quadratic least-squares system much
    # better conditioned than fitting directly to small normalized values.
    gaze_values = np.asarray(gaze_samples, dtype=np.float64)
    center = np.mean(gaze_values, axis=0)
    scale = np.maximum(np.std(gaze_values, axis=0), 0.01)
    X = np.array([
        polynomial_features((g - center) / scale)
        for g in gaze_values
    ])

    Y = np.array(screen_targets, dtype=np.float64)

    coefficients, _, _, _ = np.linalg.lstsq(
        X, Y, rcond=None
    )

    return {"coefficients": coefficients, "center": center, "scale": scale}


def predict_cursor(gaze, coefficients):
    normalized_gaze = (np.asarray(gaze) - coefficients["center"]) / coefficients["scale"]
    result = polynomial_features(normalized_gaze) @ coefficients["coefficients"]

    x = int(np.clip(result[0], 0, SCREEN_WIDTH - 1))
    y = int(np.clip(result[1], 0, SCREEN_HEIGHT - 1))

    return x, y


def summarize_calibration_samples(samples):
    """Remove landmark spikes, then measure stability of the central data.

    A raw min-to-max range lets one bad Face Mesh frame reject an otherwise
    usable target.  Per-axis MAD supplies a robust scale, while the 10th--90th
    percentile span still detects real gaze movement during the target hold.
    """
    values = np.asarray(samples, dtype=np.float64)
    median = np.median(values, axis=0)
    absolute_deviation = np.abs(values - median)
    mad = np.median(absolute_deviation, axis=0)
    robust_scale = np.maximum(1.4826 * mad, CALIBRATION_NOISE_FLOOR)
    robust_z = absolute_deviation / robust_scale
    inliers = np.all(robust_z <= CALIBRATION_OUTLIER_Z, axis=1)
    filtered = values[inliers]

    if len(filtered) == 0:
        return None, {"inliers": 0, "total": len(values), "mad": mad,
                      "central_spread": np.array([np.inf, np.inf])}

    representative = np.median(filtered, axis=0)
    central_spread = np.percentile(filtered, 90, axis=0) - np.percentile(
        filtered, 10, axis=0)
    return representative, {
        "inliers": len(filtered),
        "total": len(values),
        "mad": mad,
        "central_spread": central_spread,
    }


def calibration_samples_are_stable(summary):
    enough_inliers = summary["inliers"] >= MIN_CALIBRATION_SAMPLES
    enough_inliers &= (summary["inliers"] / summary["total"] >=
                       MIN_CALIBRATION_INLIER_RATIO)
    central_stable = float(np.max(summary["central_spread"])) <= (
        MAX_CALIBRATION_CENTRAL_SPREAD)
    mad_stable = float(np.max(summary["mad"])) <= MAX_CALIBRATION_MAD
    return enough_inliers and central_stable and mad_stable


def calibration_targets_are_distinct(gaze_samples):
    """Ensure target medians span a two-dimensional, identifiable gaze space."""
    values = np.asarray(gaze_samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        return False, np.inf, np.inf, None, None, "invalid or non-finite gaze features"
    # Dimensionless distance: scale both axes by the measured calibration
    # range so an axis's units/magnitude cannot hide separation on the other.
    scale = np.ptp(values, axis=0)
    if np.any(scale < 1e-4):
        return False, 0.0, np.inf, None, None, "one gaze axis did not change across targets"
    normalized = (values - np.mean(values, axis=0)) / scale
    distances = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    minimum_distance = float(np.min(distances))
    nearest = np.unravel_index(np.argmin(distances), distances.shape)
    center = np.mean(values, axis=0)
    feature_scale = np.maximum(np.std(values, axis=0), 0.01)
    design = np.array([polynomial_features((gaze - center) / feature_scale)
                       for gaze in values])
    condition = float(np.linalg.cond(design))
    rank = int(np.linalg.matrix_rank(design))
    reasons = []
    if minimum_distance < MIN_TARGET_GAZE_DISTANCE:
        reasons.append(f"nearest normalized pair distance {minimum_distance:.3f} < {MIN_TARGET_GAZE_DISTANCE:.3f}")
    if rank < 6:
        reasons.append(f"quadratic design rank {rank} < 6")
    if not math.isfinite(condition) or condition > MAX_CALIBRATION_CONDITION:
        reasons.append(f"design condition {condition:.1f} exceeds {MAX_CALIBRATION_CONDITION}")
    return (not reasons), minimum_distance, condition, distances, nearest, "; ".join(reasons)


def calibrate(cap, face_mesh):
    print("\nStarting nine-point calibration.")
    print("Look directly at each target.")
    print("Keep your head steady and your eyes open.")

    target_fractions = [
        (0.10, 0.15), (0.50, 0.15), (0.90, 0.15),
        (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
        (0.10, 0.85), (0.50, 0.85), (0.90, 0.85),
    ]
    target_pixels = [(int(tx * (SCREEN_WIDTH - 1)), int(ty * (SCREEN_HEIGHT - 1)))
                     for tx, ty in target_fractions]

    gaze_samples = []
    screen_targets = []

    canvas = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8,
    )

    cv2.setWindowProperty(
        WINDOW,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    cv2.waitKey(1)
    eye_summaries = []
    for number, (target_x, target_y) in enumerate(target_pixels, start=1):

        attempt = 0
        while True:
            attempt += 1
            started = time.time()
            samples = []
            left_eye_samples = []
            right_eye_samples = []
            manual_retry = False
            rejected = {"camera": 0, "face": 0, "iris": 0,
                        "closed": 0, "nonfinite": 0, "warmup": 0}

            while time.time() - started < CALIBRATION_HOLD:
                frame, gaze, ear, face, tracking_reason, quality = read_face(cap, face_mesh)

                if frame is None:
                    rejected["camera"] += 1
                    continue

                elapsed = time.time() - started

                canvas[:] = 0

                cv2.putText(
                canvas,
                f"CALIBRATION {number}/9",
                (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

                cv2.putText(
                canvas,
                "Look at the red target and keep your eyes open",
                (35, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
                cv2.putText(canvas, "R: retry this target | Q/ESC: cancel",
                            (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (200, 200, 200), 1)

                cv2.circle(
                canvas,
                (target_x, target_y),
                25,
                (0, 0, 255),
                -1,
            )

                cv2.circle(
                canvas,
                (target_x, target_y),
                38,
                (255, 255, 255),
                3,
            )

                cv2.putText(
                canvas,
                f"Valid samples: {len(samples)}",
                (35, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

                # The inset lets the user correct face position, eyelid
                # closure, or landmark loss without leaving calibration.
                inset_w = min(360, SCREEN_WIDTH // 3)
                inset_h = int(inset_w * frame.shape[0] / frame.shape[1])
                inset = cv2.resize(draw_tracking_landmarks(frame, face),
                                   (inset_w, inset_h))
                inset_x = SCREEN_WIDTH - inset_w - 18
                inset_y = 18
                canvas[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w] = inset
                cv2.rectangle(canvas, (inset_x, inset_y),
                              (inset_x + inset_w, inset_y + inset_h), (0, 255, 0), 2)

                diagnostic = tracking_reason
                if gaze is not None and ear is not None:
                    diagnostic += f" | EAR {ear:.3f} | gaze {gaze[0]:.3f}, {gaze[1]:.3f}"
                    diagnostic += (f" | lid={quality.get('lid_ratios', [0, 0])[0]:.2f}/"
                                   f"{quality.get('lid_ratios', [0, 0])[1]:.2f}")
                cv2.putText(canvas, diagnostic[:110], (35, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

                cv2.imshow(WINDOW, canvas)

            # IMPORTANT:
            # Collect valid gaze data even if EAR varies slightly.
            # Only reject missing gaze or clearly closed eyes.
                if elapsed < CALIBRATION_WARMUP:
                    rejected["warmup"] += 1
                elif gaze is None:
                    rejected["face" if tracking_reason == "Face not detected" else "iris"] += 1
                elif ear is None or not math.isfinite(ear) or not np.all(np.isfinite(gaze)):
                    rejected["nonfinite"] += 1
                elif ear < CALIBRATION_MIN_EAR:
                    rejected["closed"] += 1
                else:
                    samples.append(gaze.copy())
                    eye_features = quality.get("eye_features", [])
                    if len(eye_features) == 2:
                        left_eye_samples.append(eye_features[0])
                        right_eye_samples.append(eye_features[1])

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord("r"):
                    manual_retry = True
                    break

            representative, summary = summarize_calibration_samples(samples) if samples else (
                None, {"inliers": 0, "total": 0, "mad": np.array([np.inf, np.inf]),
                       "central_spread": np.array([np.inf, np.inf])})
            if manual_retry:
                print(f"  Retrying target {number} at your request; completed targets are kept.")
                continue
            print(f"Target {number}/9, attempt {attempt}: {len(samples)} valid samples")
            print(f"  displayed target pixel: ({target_x}, {target_y})")
            print("  rejected: " + ", ".join(
                f"{name}={count}" for name, count in rejected.items() if count))
            print("  robust data: "
                  f"inliers={summary['inliers']}/{summary['total']}, "
                  f"MAD=({summary['mad'][0]:.3f}, {summary['mad'][1]:.3f}), "
                  f"P10-P90=({summary['central_spread'][0]:.3f}, "
                  f"{summary['central_spread'][1]:.3f})")

            left_median = np.median(left_eye_samples, axis=0) if left_eye_samples else np.array([np.nan, np.nan])
            right_median = np.median(right_eye_samples, axis=0) if right_eye_samples else np.array([np.nan, np.nan])
            if representative is not None:
                print(f"  median gaze H/V: ({representative[0]:.5f}, {representative[1]:.5f})")
                print(f"  left-eye H/V: ({left_median[0]:.5f}, {left_median[1]:.5f}); right-eye H/V: ({right_median[0]:.5f}, {right_median[1]:.5f})")
            if representative is not None and calibration_samples_are_stable(summary):
                gaze_samples.append(representative)
                screen_targets.append((target_x, target_y))
                eye_summaries.append((left_median, right_median))
                break

            if attempt >= MAX_TARGET_ATTEMPTS:
                raise RuntimeError(
                    f"Target {number} remained unstable after {attempt} attempts. "
                    "Keep your head still, look only at the target, and improve lighting."
                )
            print(f"  Retrying target {number}: central gaze variation is too high or "
                  "too many frames were outliers. Hold your gaze steady.")

    distinct, minimum_distance, condition, pair_distances, nearest, reject_reason = calibration_targets_are_distinct(gaze_samples)
    print("\nCalibration target medians (screen pixels; eye-relative H/V):")
    for i, ((sx, sy), gaze, eyes) in enumerate(zip(screen_targets, gaze_samples, eye_summaries), start=1):
        left, right = eyes
        print(f"  T{i} screen=({sx},{sy}) combined=({gaze[0]:.5f},{gaze[1]:.5f}) left=({left[0]:.5f},{left[1]:.5f}) right=({right[0]:.5f},{right[1]:.5f})")
    if pair_distances is not None:
        print("  Pair distances (range-normalized H/V features):")
        for i in range(len(gaze_samples)):
            row = ["  --  " if i == j else f"{pair_distances[i, j]:.3f}" for j in range(len(gaze_samples))]
            print(f"  T{i + 1}: " + " ".join(row))
        a, b = nearest
        print(f"  Nearest pair: T{a + 1} <-> T{b + 1}: {minimum_distance:.4f}")
    if not distinct:
        raise RuntimeError(
            "Calibration rejected: gaze features did not distinguish the displayed targets. "
            f"Reason: {reject_reason}. Design condition: {condition:.1f}. "
            "Use the printed medians to check whether gaze features changed as you followed each target. "
            "Verify refined iris tracking, lighting, and eye-only target following."
        )

    coefficients = fit_calibration(
        gaze_samples,
        screen_targets,
    )

    cv2.setWindowProperty(
        WINDOW,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW,
        PREVIEW_WIDTH,
        PREVIEW_HEIGHT,
    )

    cv2.moveWindow(WINDOW, 20, 20)

    print("Calibration completed successfully.")

    return coefficients


# ==========================================================
# 5. MOUSE CONTROL
# ==========================================================

def run_cursor_control(cap, face_mesh, coefficients):
    print("\nEye cursor control is active.")
    print("Look to move the cursor.")
    print("Hold gaze still to dwell-click.")
    print("Brief deliberate blink to blink-click.")
    print("Q/ESC: Quit | C: Recalibrate | SPACE: Pause")

    smooth_x = None
    smooth_y = None

    dwell_anchor = None
    dwell_start = time.time()
    dwell_armed = True

    eyes_closed_since = None
    blink_ignored = False

    paused = False

    while True:
        frame, gaze, ear, face, tracking_reason, _quality = read_face(cap, face_mesh)

        if frame is None:
            continue

        now = time.time()
        status = "ACTIVE"
        dwell_progress = 0.0

        if gaze is None or ear is None:
            status = tracking_reason

            # Do not count loss of tracking as a blink.
            eyes_closed_since = None
            blink_ignored = False

            dwell_anchor = None
            dwell_start = now

        elif paused:
            status = "PAUSED - press SPACE to resume"

            eyes_closed_since = None
            blink_ignored = False

        else:
            eyes_open = ear >= BLINK_THRESHOLD
            blink_clicked = False

            # ----------------------------------------------
            # BLINK CLICK
            # ----------------------------------------------
            if not eyes_open:
                if eyes_closed_since is None:
                    eyes_closed_since = now

                duration = now - eyes_closed_since

                if duration > MAX_BLINK_TIME:
                    blink_ignored = True
                    status = "Long eye closure ignored"
                else:
                    status = "Eyes closed"

            else:
                # Trigger only when the eyes reopen after
                # a short closure.
                if eyes_closed_since is not None:
                    duration = now - eyes_closed_since

                    if (
                        MIN_BLINK_TIME <= duration <= MAX_BLINK_TIME
                        and not blink_ignored
                    ):
                        try:
                            pyautogui.click()
                            blink_clicked = True
                            status = "BLINK CLICK!"

                        except pyautogui.FailSafeException:
                            print("Emergency stop triggered.")
                            return

                    eyes_closed_since = None
                    blink_ignored = False

                # ------------------------------------------
                # CURSOR MOVEMENT
                # ------------------------------------------
                raw_x, raw_y = predict_cursor(
                    gaze, coefficients
                )

                if smooth_x is None:
                    smooth_x = float(raw_x)
                    smooth_y = float(raw_y)
                else:
                    smooth_x = (
                        SMOOTHING * raw_x
                        + (1.0 - SMOOTHING) * smooth_x
                    )

                    smooth_y = (
                        SMOOTHING * raw_y
                        + (1.0 - SMOOTHING) * smooth_y
                    )

                cursor_x = int(np.clip(
                    smooth_x, 0, SCREEN_WIDTH - 1
                ))

                cursor_y = int(np.clip(
                    smooth_y, 0, SCREEN_HEIGHT - 1
                ))

                try:
                    pyautogui.moveTo(
                        cursor_x, cursor_y, duration=0
                    )
                except pyautogui.FailSafeException:
                    print("Emergency stop triggered.")
                    return

                current_position = (cursor_x, cursor_y)

                # ------------------------------------------
                # DWELL CLICK
                # ------------------------------------------
                if dwell_anchor is None:
                    dwell_anchor = current_position
                    dwell_start = now

                movement = distance(
                    current_position, dwell_anchor
                )

                if blink_clicked:
                    # Avoid a dwell click immediately after
                    # a blink click.
                    dwell_anchor = current_position
                    dwell_start = now
                    dwell_armed = True

                elif not dwell_armed:
                    if movement >= DWELL_REARM_DISTANCE:
                        dwell_armed = True
                        dwell_anchor = current_position
                        dwell_start = now

                    status = "Move gaze to re-arm dwell-click"

                elif movement > DWELL_RADIUS:
                    dwell_anchor = current_position
                    dwell_start = now
                    status = "Tracking eyes"

                else:
                    elapsed = now - dwell_start

                    dwell_progress = min(
                        elapsed / DWELL_TIME, 1.0
                    )

                    if elapsed >= DWELL_TIME:
                        try:
                            pyautogui.click()
                            dwell_armed = False
                            dwell_anchor = current_position
                            status = "DWELL CLICK!"
                        except pyautogui.FailSafeException:
                            print("Emergency stop triggered.")
                            return
                    else:
                        status = (
                            f"Dwell click in "
                            f"{DWELL_TIME - elapsed:.1f}s"
                        )

                # Gaze indicator on the preview.
                px = int(
                    cursor_x / SCREEN_WIDTH * PREVIEW_WIDTH
                )

                py = int(
                    cursor_y / SCREEN_HEIGHT * PREVIEW_HEIGHT
                )

                px = max(0, min(PREVIEW_WIDTH - 1, px))
                py = max(0, min(PREVIEW_HEIGHT - 1, py))

                cv2.circle(
                    frame,
                    (px, py),
                    12,
                    (0, 255, 0),
                    2,
                )

        show_preview(
            frame,
            gaze,
            ear,
            status,
            dwell_progress,
            face=face,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            paused = not paused

            eyes_closed_since = None
            blink_ignored = False

            dwell_anchor = None
            dwell_start = time.time()

        elif key == ord("c"):
            coefficients = calibrate(cap, face_mesh)

            smooth_x = None
            smooth_y = None

            dwell_anchor = None
            dwell_start = time.time()
            dwell_armed = True

            eyes_closed_since = None
            blink_ignored = False

    print("Eye cursor stopped.")


# ==========================================================
# 6. MAIN
# ==========================================================

def main():
    cap = None

    try:
        cap = open_camera()

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            WINDOW,
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
        )

        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as face_mesh:

            print("Initializing webcam and eye tracking...")

            # Warm up the camera and face detector.
            for _ in range(30):
                read_face(cap, face_mesh)

            coefficients = calibrate(cap, face_mesh)

            run_cursor_control(
                cap,
                face_mesh,
                coefficients,
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as error:
        print(f"\nERROR: {error}")

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        print("Webcam released.")
        print("Program closed.")


if __name__ == "__main__":
    main()
