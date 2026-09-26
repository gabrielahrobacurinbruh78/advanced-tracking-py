from dataclasses import dataclass
import random
import time
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PipelineConfig:
    # Everything the pipeline tunes lives here so a single construction call can reconfigure it.
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    camera_fps: int = 60
    pinch_threshold_px: float = 60.0
    filter_cooldown_sec: float = 0.15
    mode_cooldown_sec: float = 1.2
    fist_dist_threshold_px: float = 80.0
    third_portal_enabled: bool = True
    third_portal_style: str = "hand"
    third_portal_lock_style: bool = True
    third_portal_height_scale: float = 1.0
    third_portal_min_height_px: int = 24
    third_portal_max_height_frac: float = 0.6
    third_portal_gap_px: int = 12


class FilterBank:
    # Per-portal looks. Every filter takes the portal's own ROI, so the cost stays proportional
    # to pane area instead of frame area; nothing here may touch global state.

    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        # BGR, not RGB: (10, 140, 255) reads orange and (180, 30, 220) reads magenta on screen.
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        # Color-dodge by division: bright where the blurred inverse is dark, i.e. pencil strokes.
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        # Down-then-up with nearest neighbour: the blocky look costs two cheap resizes.
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def glitch(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        # Rolling red and blue in opposite directions is the cheapest chromatic split available.
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            # Two random rows of pure noise read as analogue static without a second pass.
            y = random.randint(0, h - 1)
            out[y] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        _, _, r = cv2.split(roi)
        zeros = np.zeros_like(r)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def edge(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        # Mask with the edge image itself so the background stays black instead of tinted.
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(roi, (25, 25), 0)

    @staticmethod
    def cartoon(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        # Bilateral keeps flat colour regions while the adaptive threshold supplies the ink lines.
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        t = time.time() * 5.0
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x + y) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        # Weighted blend keeps the camera readable under the moving wash; the phase makes it animate.
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)


class GeometryUtils:
    # Small point maths shared by the gesture tests and the portal builders. Kept separate from
    # PortalProcessor so the shapes can be reasoned about (and unit-checked) without a camera.

    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def polygon_vertical_span(pts: List[Tuple[int, int]]) -> Tuple[int, int]:
        ys = np.asarray(pts, dtype=np.int32)[:, 1]
        return (int(ys.min()), int(ys.max()))

    @staticmethod
    def is_fist_closed(landmarks: List[object], w: int, h: int, threshold: float) -> bool:
        wrist = np.array([landmarks[0].x * w, landmarks[0].y * h], dtype=np.float64)
        # All four fingertips bunched around the wrist at once is the only fist signal that
        # survives the rotated, mirrored poses this pipeline sees.
        distances = [float(np.linalg.norm(np.array([landmarks[i].x * w, landmarks[i].y * h]) - wrist))
                     for i in (8, 12, 16, 20)]
        return float(np.mean(distances)) < threshold

    @staticmethod
    def is_hand_rotated(thumb: Tuple[int, int], index: Tuple[int, int]) -> bool:
        dx = float(index[0] - thumb[0])
        dy = float(index[1] - thumb[1])
        # Either the index tip sits well below the thumb, or the thumb-index vector is mostly
        # horizontal: both mean the quad built from raw tips would self-intersect.
        return bool(dy > 25 or abs(dx) > abs(dy) * 1.1)

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.int32)
        ordered = arr[np.argsort(arr[:, 0])]
        left = ordered[:2]
        right = ordered[2:]
        left = left[np.argsort(left[:, 1])]
        right = right[np.argsort(right[:, 1])]
        left_top, left_bottom = left[0], left[1]
        right_top, right_bottom = right[0], right[1]
        return np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.int32)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.int32)
        ordered = arr[np.argsort(arr[:, 0])]
        left = ordered[:2]
        right = ordered[2:]
        left = left[np.argsort(left[:, 1])]
        right = right[np.argsort(right[:, 1])]
        left_top, left_bottom = left[0], left[1]
        right_top, right_bottom = right[0], right[1]
        # Deliberately twisted order: a rotated hand fills the bowtie region instead of leaving
        # two triangles empty.
        return np.array([left_top, right_bottom, right_top, left_bottom], dtype=np.int32)


class PortalProcessor:
    # Owns the mediapipe graph, the filter bank and all per-session state. process_frame is the
    # only entry point used by the render loop; nothing accumulates between frames.

    THIRD_PORTAL_STYLES = ("hand", "mirror", "stack")
    THIRD_PORTAL_DEFAULT_STYLE = THIRD_PORTAL_STYLES[0]

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        # Insertion order is the cycle order: pinch / N / P walk this dict front to back.
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "dual-tone": FilterBank.dual_tone,
            "thermal": FilterBank.thermal,
            "sketch": FilterBank.sketch,
            "pixelate": FilterBank.pixelate,
            "glitch": FilterBank.glitch,
            "invert": FilterBank.invert,
            "red-channel": FilterBank.red_channel,
            "edge": FilterBank.edge,
            "blur": FilterBank.blur,
            "cartoon": FilterBank.cartoon,
            "rainbow-wave": FilterBank.rainbow_wave,
        }
        self.filter_keys = list(self.filters.keys())
        self.active_filter_idx = 0
        self.is_3d_mode = True

        self.third_portal_enabled = cfg.third_portal_enabled
        self.third_portal_lock_style = cfg.third_portal_lock_style
        # A locked processor ignores the configured style: the default is the only supported one.
        self.third_portal_style = (self.THIRD_PORTAL_DEFAULT_STYLE if self.third_portal_lock_style
                                   else cfg.third_portal_style)

        self.last_switch_time = 0.0
        self.last_mode_toggle = 0.0
        self.last_measured_fps = 0.0

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=0.3, min_tracking_confidence=0.3)

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    @property
    def secondary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 1) % len(self.filter_keys)]

    @property
    def tertiary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 2) % len(self.filter_keys)]

    @property
    def current_mode_name(self) -> str:
        return "3D" if self.is_3d_mode else "2D"

    def cycle_filter(self, step: int = 1) -> None:
        # No cooldown here: the gesture handler owns the rate limit, keys are already discrete.
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)

    def cycle_third_portal_style(self) -> None:
        if self.third_portal_lock_style:
            # Locked: Y stays inert and any drift snaps back to the default style.
            self.third_portal_style = self.THIRD_PORTAL_DEFAULT_STYLE
            return
        try:
            idx = self.THIRD_PORTAL_STYLES.index(self.third_portal_style)
        except ValueError:
            idx = -1  # unknown style -> the next index is 0, i.e. "hand"
        self.third_portal_style = self.THIRD_PORTAL_STYLES[(idx + 1) % len(self.THIRD_PORTAL_STYLES)]

    def toggle_mode(self) -> None:
        now = time.time()
        # Held fists fire every frame; the cooldown turns that into one toggle per gesture.
        if now - self.last_mode_toggle > self.cfg.mode_cooldown_sec:
            self.is_3d_mode = not self.is_3d_mode
            self.last_mode_toggle = now

    def render_portal(self, frame: np.ndarray, pts: List[Tuple[int, int]], filter_key: str) -> np.ndarray:
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
        if w <= 10 or h <= 10:
            # Collapsed panes and hands fully outside the frame end here; this is the single
            # guard that keeps the rest of the pipeline free of bounds checks.
            return frame
        roi = frame[y:y + h, x:x + w].copy()
        processed_roi = self.filters[filter_key](roi)
        mask = np.zeros((h, w), np.uint8)
        # Translate into ROI space only. Points beyond the frame are clipped by fillPoly rather
        # than rescaled, so a pane hanging off an edge is cut, never distorted.
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        # One add plus two bitwise_and beats merging the mask to three channels and running a
        # per-channel blend; both operands are already ROI-sized.
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        frame[y:y + h, x:x + w] = cv2.add(bg, fg)
        # Every pane is outlined so adjacent panes stay readable when their filters match.
        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
        return frame

    def render_portal_below(
            self, frame: np.ndarray, base_pts: List[Tuple[int, int]], filter_key: str,
            band_ends: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None) -> np.ndarray:
        if self.third_portal_style == "hand" and band_ends:
            pts: Optional[List[Tuple[int, int]]] = self._third_portal_from_hands(frame, band_ends)
        else:
            pts = self._third_portal_copied_from(frame, base_pts)
        if pts is None:
            # No finger data or no room below portal #2: leave the frame exactly as it is.
            return frame
        return self.render_portal(frame, pts, filter_key)

    def _third_portal_from_hands(
            self, frame: np.ndarray,
            band_ends: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[int, int]]:
        min_h = max(4, int(self.cfg.third_portal_min_height_px))
        frac = min(max(float(self.cfg.third_portal_max_height_frac), 0.1), 1.0)
        max_h = max(min_h, int(frame.shape[0] * frac))
        scale = float(self.cfg.third_portal_height_scale)

        tops: List[Tuple[int, int]] = []
        bottoms: List[Tuple[int, int]] = []
        for ring, pinky in band_ends:
            # Tips arrive as frame pixels (see process_frame); the getattr fallback also accepts
            # landmark-like objects so both callers stay in one coordinate space.
            rx = float(getattr(ring, "x", ring[0]))
            ry = float(getattr(ring, "y", ring[1]))
            qx = float(getattr(pinky, "x", pinky[0]))
            qy = float(getattr(pinky, "y", pinky[1]))
            dx = qx - rx
            dy = qy - ry
            dist = float(np.hypot(dx, dy))
            if dist < 1e-6:
                # Ring and pinky on the same pixel: straight down keeps the pane well defined.
                ux, uy = 0.0, 1.0
                run = float(min_h)
            else:
                ux, uy = dx / dist, dy / dist
                run = dist * scale
            h = min(max(run, float(min_h)), float(max_h))
            tops.append((int(round(rx)), int(round(ry))))
            bottoms.append((int(round(rx + ux * h)), int(round(ry + uy * h))))
        # Built from the landmarks, so the pane leans, grows and travels with the hands and shares
        # portal #2's ring-tip edge exactly. Hand #1 is the left pair.
        return [tops[0], bottoms[0], bottoms[1], tops[1]]

    def _third_portal_copied_from(self, frame: np.ndarray,
                                 base_pts: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        top_y, bottom_y = GeometryUtils.polygon_vertical_span(base_pts)
        span = bottom_y - top_y
        fold_y = bottom_y + max(0, int(self.cfg.third_portal_gap_px))
        room = (frame.shape[0] - 1) - fold_y
        if span <= 0 or room <= 10:
            # Degenerate base polygon, or portal #2 already sits on the bottom edge.
            return None
        # Squashing guarantees the copy fits; the hand style deliberately does not squash.
        scale_y = min(1.0, room / span)
        if self.third_portal_style == "stack":
            # Repeat the shape lower down, same orientation.
            return [(int(x), int(fold_y + (y - top_y) * scale_y)) for x, y in base_pts]
        # "mirror": fold the shape down about its own lower edge so the shared edge lines up.
        return [(int(x), int(fold_y + (bottom_y - y) * scale_y)) for x, y in base_pts]

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        # Resize first, mirror second: identical pixels, but the flip runs on the 960x540 working
        # frame instead of the full-size camera buffer.
        frame = cv2.resize(frame, (cfg.frame_width, cfg.frame_height))
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Read-only input: mediapipe skips its internal copy when the buffer is not writeable.
        rgb.flags.writeable = False

        def to_frame_pt(x_norm: float, y_norm: float) -> Tuple[int, int]:
            return (int(round(x_norm * cfg.frame_width)), int(round(y_norm * cfg.frame_height)))

        results = self.hands.process(rgb)
        now = time.time()

        all_hand_tips: List[List[Tuple[int, int]]] = []
        is_bowtie = False
        fist_count = 0

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                # Real mediapipe nests the 21 points under `.landmark`; a bare sequence works too,
                # which is what lets the pipeline be driven headlessly.
                hand_lm = getattr(hand, "landmark", hand)
                # Landmark coordinates are used exactly as mediapipe returns them: the mirror is
                # applied to pixels only, so a point is never x-flipped.
                points = [to_frame_pt(lm.x, lm.y) for lm in hand_lm]
                for pt in points:
                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                             (18, 19), (19, 20), (0, 17)):
                    cv2.line(frame, points[i], points[j], (0, 255, 0), 2)

                # thumb, index, middle, ring, pinky
                tips = [to_frame_pt(hand_lm[i].x, hand_lm[i].y) for i in (4, 8, 12, 16, 20)]
                all_hand_tips.append(tips)

                # Thumb touching pinky or ring is the filter cycle; the cooldown absorbs the fact
                # that a held pinch reports on every frame.
                pinched = (GeometryUtils.euclidean_dist(tips[0], tips[4]) < cfg.pinch_threshold_px
                           or GeometryUtils.euclidean_dist(tips[0], tips[3]) < cfg.pinch_threshold_px)
                if pinched and now - self.last_switch_time > cfg.filter_cooldown_sec:
                    self.cycle_filter(1)
                    self.last_switch_time = now

                if GeometryUtils.is_fist_closed(hand_lm, cfg.frame_width, cfg.frame_height,
                                                cfg.fist_dist_threshold_px):
                    fist_count += 1

        # Two fisted hands at once is the 3D/2D switch, rate limited by the same reasoning.
        if fist_count >= 2 and now - self.last_mode_toggle > self.cfg.mode_cooldown_sec:
            self.is_3d_mode = not self.is_3d_mode
            self.last_mode_toggle = now

        if self.is_3d_mode:
            if len(all_hand_tips) == 2:
                t1, t2 = all_hand_tips
                # #1 and #2 share the middle-tip edge, #2 and #3 share the ring-tip edge: painting
                # in this order leaves no seam and no overlap between the three bands.
                first_portal = [t1[0], t1[1], t1[2], t2[2], t2[1], t2[0]]
                second_portal = [t1[2], t1[3], t2[3], t2[2]]
                frame = self.render_portal(frame, first_portal, self.current_filter_name)
                frame = self.render_portal(frame, second_portal, self.secondary_filter_name)
                if self.third_portal_enabled:
                    frame = self.render_portal_below(frame, second_portal, self.tertiary_filter_name,
                                                     band_ends=[(t1[3], t1[4]), (t2[3], t2[4])])
            elif len(all_hand_tips) == 1:
                # One hand cannot define a second band, so portals #2 and #3 do not exist here.
                frame = self.render_portal(frame, all_hand_tips[0], self.current_filter_name)
        else:
            if len(all_hand_tips) == 2:
                t1, t2 = all_hand_tips
                corners = [t1[0], t1[1], t2[0], t2[1]]
                if GeometryUtils.is_hand_rotated(t1[0], t1[1]) or GeometryUtils.is_hand_rotated(t2[0], t2[1]):
                    quad = GeometryUtils.sort_quad_bowtie(corners)
                    is_bowtie = True
                else:
                    quad = GeometryUtils.sort_quad_clean(corners)
                frame = self.render_portal(frame, quad, self.current_filter_name)
            elif len(all_hand_tips) == 1:
                tips = all_hand_tips[0]
                quad = GeometryUtils.sort_quad_clean([tips[0], tips[1], tips[2], tips[4]])
                frame = self.render_portal(frame, quad, self.current_filter_name)

        self._draw_hud(frame, is_bowtie)
        return frame

    def _draw_hud(self, frame: np.ndarray, is_bowtie: bool) -> None:
        # is_bowtie is accepted for the rotated-hand case but currently draws nothing extra; the
        # parameter stays so the HUD signature does not change when that text is added.
        third_label = "ON" if self.third_portal_enabled else "OFF"
        style_hint = "locked" if self.third_portal_lock_style else "[Y] style"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"PORTAL MODE: {self.current_mode_name}", (15, 25), font, 0.55, (255, 0, 0), 2)
        cv2.putText(frame, "Press M to toggle 3D / 2D", (15, 48), font, 0.45, (0, 200, 255), 1)
        cv2.putText(frame, f"FPS: {self.last_measured_fps:.0f}", (15, 71), font, 0.45, (0, 255, 0), 1)
        cv2.putText(frame,
                    f"P3 {self.tertiary_filter_name} ({self.third_portal_style}, {style_hint}) "
                    f"size x{self.cfg.third_portal_height_scale:.2f} under P2: {third_label}",
                    (15, 94), font, 0.42, (0, 255, 255), 1)


def main() -> None:
    cfg = PipelineConfig()
    processor = PortalProcessor(cfg)

    # DirectShow opens faster and has lower latency on Windows.
    cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Not every platform ships DirectShow; fall back to the OpenCV default backend.
        cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened():
        print("[ERROR] fix it.")
        return

    # MJPG is sent uncompressed over USB, which is what makes 960x540 at 60 fps reachable.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)  # match the working resolution exactly
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)  # ...so no per-frame upscale or crop
    cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)  # requested capture rate; MJPG makes it attainable
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # no stale buffered frames -> lower latency

    fps = 0.0
    last_fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] bro fix your code.")
            break

        out_frame = processor.process_frame(frame)
        cv2.imshow("3d mesh hyperengine", out_frame)

        frame_count += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = frame_count / (now - last_fps_time)
            frame_count = 0
            last_fps_time = now
        # Published every frame, not only on recompute, so the HUD always shows the last
        # full-second measurement instead of a value that only refreshes once per second.
        processor.last_measured_fps = fps

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("n"):
            processor.cycle_filter(1)
        elif key == ord("p"):
            processor.cycle_filter(-1)
        elif key in (ord("m"), ord("M")):
            processor.toggle_mode()
        elif key in (ord("y"), ord("Y")):
            processor.cycle_third_portal_style()  # inert while the style is locked
        elif key == ord("s"):
            cv2.imwrite(f"cap_{int(time.time())}.png", out_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
