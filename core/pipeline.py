"""
core.pipeline
=============
VideoPipelineMixin: the real-time capture -> detect -> identify -> track
-> render pipeline. Three background threads (video/render/detect) plus
the Qt-side frame poll and recognition-queue drain that run on the main
thread. Mixed into FaceRecognitionApp (see userinterface.app) alongside the other
tab-specific mixins.
"""

import time
import queue
import threading

# pyrefly: ignore [missing-import]
from datetime import date, datetime
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PyQt6.QtCore import QTimer
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QPixmap

import logging
log = logging.getLogger("FaceRecog")

from services import database
from services import encoder
from services.tts import speak_greeting

from .config import (
    CONFIDENCE_THRESHOLD, IDENTITY_MARGIN, ASSOC_EMBED_VETO_SIM,
    DETECT_EVERY_N_FRAMES, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT, _RENDER_INTERVAL,
    _CAMERA_IDLE_SLEEP, REID_SIMILARITY_THRESHOLD, REID_MARGIN,
)
from .tracking import SmoothBox, _iou, _center_dist, _draw_corner_tracers, _draw_scan_line
from .camera_stream import ThreadedCamera


class VideoPipelineMixin:
    """Capture/track/detect/render loops + recognition-queue handling.

    Expects the host class (FaceRecognitionApp) to provide the Qt widgets
    (video_label, status_bar, camera_var, colors, ...) and the shared
    state initialized in FaceRecognitionApp.__init__ (self.cap,
    self._boxes, self._enc_lock, self.known_encodings, etc.).
    """

    def _start_threads(self):
        opened_idx = 1
        for idx in [1, 0, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.release()
                    self.cap = ThreadedCamera(idx)
                    opened_idx = idx
                    log.info(f"Camera opened at index {idx}")
                    break
                else:
                    cap.release()

        if not self.cap or not self.cap.isOpened():
            self.status_bar.setText("● Camera not found")
            self.status_bar.setStyleSheet(
                f"color: {self.colors['danger']}; background: transparent;")
            log.error("No camera found at indices 0-3")
            return

        self.camera_var.set(f"Camera {opened_idx}")
        self.running = True
        threading.Thread(target=self._video_loop,  name="VideoThread",  daemon=True).start()
        threading.Thread(target=self._render_loop, name="RenderThread", daemon=True).start()
        threading.Thread(target=self._detect_loop, name="DetectThread", daemon=True).start()

        # PERF FIX: _poll_frame used to re-arm itself via self.after(1/2, ...)
        # every tick — at ~500-1000 calls/sec that created and (even after
        # the after() leak fix) tore down hundreds of QTimer objects per
        # second for no reason. One persistent repeating QTimer does the
        # same job with a single long-lived object.
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_frame)
        self._poll_timer.start(1)

    def _change_camera(self, choice):
        try:
            idx = int(choice.split(" ")[1])
            self._requested_camera_idx = idx
        except Exception as e:
            log.error(f"Failed to parse camera choice {choice}: {e}")

    def _video_loop(self):
        while self.running:
            if self._requested_camera_idx is not None:
                if self.cap:
                    self.cap.release()
                self.cap = ThreadedCamera(self._requested_camera_idx)
                self._requested_camera_idx = None

            if not self.cap or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ret, cpu_frame = self.cap.read()
            if not ret or cpu_frame is None:
                time.sleep(_CAMERA_IDLE_SLEEP)
                continue

            cpu_frame = cv2.resize(cpu_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            with self._last_frame_lock:
                self._last_good_frame = cpu_frame

            with self._latest_frame_lock:
                self._latest_captured_frame = cpu_frame

            self._frame_count += 1
            if self._frame_count % DETECT_EVERY_N_FRAMES == 0:
                with self._pending_frame_lock:
                    self._pending_frame = cpu_frame.copy()
                self._detect_event.set()

    def _render_loop(self):
        next_tick = time.perf_counter()
        while self.running:
            with self._latest_frame_lock:
                cpu_frame = self._latest_captured_frame

            if cpu_frame is None:
                time.sleep(_CAMERA_IDLE_SLEEP)
                next_tick = time.perf_counter()
                continue

            fh, fw = cpu_frame.shape[:2]

            now_perf = time.perf_counter()
            dt = min(now_perf - self._last_lerp_time, 0.05)
            self._last_lerp_time = now_perf

            with self._boxes_lock:
                self._boxes = [b for b in self._boxes if not b.is_stale()]
                snapshot = list(self._boxes)

            for sb in snapshot:
                sb.lerp_dt(dt)

            frame = cpu_frame.copy()

            for sb in snapshot:
                staleness = sb.staleness()
                alpha = max(0.0, 1.0 - staleness)
                if alpha < 0.05:
                    continue

                top, right, bottom, left = sb.ints()
                top    = max(0, min(top,    fh - 1))
                bottom = max(0, min(bottom, fh - 1))
                left   = max(0, min(left,   fw - 1))
                right  = max(0, min(right,  fw - 1))
                if right <= left or bottom <= top:
                    continue

                if sb.name != "Unknown":
                    base_color = (0, 255, 130)
                    label = f"{sb.name}  {int(sb.confidence * 100)}%"
                else:
                    base_color = (50, 150, 255)
                    label = "Unknown"

                color = tuple(int(c * alpha) for c in base_color)
                thickness = 2 if alpha > 0.5 else 1

                _draw_corner_tracers(frame, left, top, right, bottom,
                                     color, thickness=thickness, alpha=alpha)
                if alpha > 0.7:
                    _draw_scan_line(frame, left, top, right, bottom,
                                    color, thickness=1)

                font_scale = 0.58
                (lw, lh), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                label_y = top - 10 if top - 10 > lh + 4 else bottom + lh + 10

                rx1 = max(0, left)
                ry1 = max(0, label_y - lh - 6)
                rx2 = min(fw, left + lw + 10)
                ry2 = min(fh, label_y + 4)
                if rx2 > rx1 and ry2 > ry1:
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)

                cv2.putText(frame, label, (left + 5, label_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Resize to display target and convert to RGB numpy for the queue
            with self._display_target_lock:
                disp_w, disp_h = self._display_target

            if (disp_w, disp_h) != (frame.shape[1], frame.shape[0]):
                interp = cv2.INTER_LINEAR if disp_w > frame.shape[1] else cv2.INTER_AREA
                out = cv2.resize(frame, (disp_w, disp_h), interpolation=interp)
            else:
                out = frame
            # Convert to contiguous RGB array — _poll_frame reads this directly
            # into QImage with zero compression, no PIL wrapper needed
            rgb = np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

            # Keep only the freshest frame in the queue
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(rgb)
            except queue.Full:
                pass

            next_tick += _RENDER_INTERVAL
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()

    def _poll_frame(self):
        if not self.running:
            return
        try:
            # Compute display target size from the video_label geometry
            cw = self.video_label.width()
            ch = self.video_label.height()
            if cw > 100 and ch > 100:
                if self._last_display_size != (cw, ch):
                    available_w = cw - 16
                    available_h = ch - 52
                    img_aspect = DISPLAY_WIDTH / DISPLAY_HEIGHT
                    test_h = available_w / img_aspect
                    if test_h <= available_h:
                        new_w, new_h = int(available_w), int(test_h)
                    else:
                        new_h = int(available_h)
                        new_w = int(available_h * img_aspect)
                    new_w = max(1, new_w)
                    new_h = max(1, new_h)
                    if new_w > MAX_DISPLAY_WIDTH or new_h > MAX_DISPLAY_HEIGHT:
                        cap_scale = min(MAX_DISPLAY_WIDTH / new_w,
                                        MAX_DISPLAY_HEIGHT / new_h)
                        new_w = max(1, int(new_w * cap_scale))
                        new_h = max(1, int(new_h * cap_scale))
                    self._last_display_size = (cw, ch)
                    self._last_render_size  = (new_w, new_h)
                    with self._display_target_lock:
                        self._display_target = (new_w, new_h)

            # Grab pre-converted RGB array and push to QLabel as QPixmap
            rgb = self._frame_queue.get_nowait()
            try:
                while True:
                    rgb = self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            # pyrefly: ignore [missing-import]
            from PyQt6.QtGui import QImage
            h_px, w_px, ch = rgb.shape
            qimg = QImage(rgb.data, w_px, h_px, w_px * ch, QImage.Format.Format_RGB888)
            px = QPixmap.fromImage(qimg.copy())
            self.video_label.setPixmap(px)

            self._fps_counter += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                detections = len(self._boxes)
                cam_fps = self.cap.fps if hasattr(self.cap, 'fps') else 0
                status_text = (
                    f"● LIVE  |  App: {self._fps_counter} FPS  "
                    f"|  Cam: {cam_fps} FPS  |  Faces: {detections}  "
                    f"|  Today: {len(self.today_logged)}"
                )
                self.status_bar.setText(status_text)
                self.status_bar.setStyleSheet(
                    f"color: {self.colors['success']}; background: transparent;")
                self._fps_counter = 0
                self._fps_timer   = now

            return

        except queue.Empty:
            pass
        except Exception as e:
            log.debug(f"Frame display error: {e}")

    def _detect_loop(self):
        while self.running:
            triggered = self._detect_event.wait(timeout=0.1)
            if not triggered or not self.running:
                continue
            self._detect_event.clear()

            with self._pending_frame_lock:
                frame = self._pending_frame
            if frame is None:
                continue

            try:
                analyzed = encoder.analyze_frame(frame)
            except Exception as e:
                log.debug(f"InsightFace analyze error: {e}")
                continue

            if not analyzed:
                continue

            fh, fw = frame.shape[:2]

            # ── Stage 1: raw-detection NMS (THE actual fix for duplicate
            # boxes on one physical face) ────────────────────────────────
            #
            # ROOT CAUSE: encoder.analyze_frame() runs InsightFace's
            # detector once per detect cycle and can legitimately return
            # MORE THAN ONE raw detection for the SAME physical face. This
            # is normal RetinaFace-style detector behavior: it proposes
            # boxes at multiple anchor scales/positions, and post-detector
            # NMS inside InsightFace itself is tuned for general detection
            # quality, not guaranteed to collapse every near-duplicate to
            # exactly one box — particularly at the IoU/confidence margins
            # this codebase already loosened (MIN_DET_SCORE=0.65, no size
            # filter) to catch long-range/angled faces. Two raw boxes for
            # one face are typically NOT pixel-identical: slightly
            # different crop bounds. That's what makes this bug so
            # specifically nasty rather than just "two boxes" — InsightFace
            # computes the embedding from the CROPPED face, so two
            # slightly different crops of the same face can yield two
            # slightly different embeddings. One crop's embedding can clear
            # CONFIDENCE_THRESHOLD/IDENTITY_MARGIN against the gallery while
            # the other's doesn't (e.g. one crop includes a hair occlusion,
            # one doesn't). Downstream, the global track-assignment step is
            # strictly 1:1 (one detection per track per cycle), so when
            # both raw detections compete for the SAME existing track, only
            # one wins it — the loser, now an apparent "new, spatially
            # unmatched detection" with its own (possibly Unknown) identity
            # result, mints a brand-new SmoothBox. That is exactly the
            # "one correct name + one Unknown" symptom: two raw detections
            # of one face, recognized inconsistently, racing for one track.
            #
            # FIX: collapse near-duplicate raw detections into one BEFORE
            # identification ever runs, using IoU-based NMS. When two raw
            # boxes overlap heavily enough that they can only plausibly be
            # the same physical face, keep only the one with the highest
            # detector confidence (det_score) and discard the rest. This
            # guarantees at most one identification pass, one embedding,
            # and one candidate box per physical face reaches the tracker —
            # which is the only point in the pipeline where this can be
            # fixed correctly, since fixing it after identification or
            # after track assignment is inherently working from already-
            # corrupted (two-embeddings-one-face) data.
            DETECTION_NMS_IOU = 0.4  # boxes overlapping more than this are treated as the same physical face

            raw_faces = []
            for face in analyzed:
                x1, y1, x2, y2 = face["bbox"]
                x1 = max(0, min(x1, fw - 1)); y1 = max(0, min(y1, fh - 1))
                x2 = max(0, min(x2, fw - 1)); y2 = max(0, min(y2, fh - 1))
                det_score = float(face.get("det_score", 1.0))
                if det_score < 0.65:  # MIN_DET_SCORE — shirt prints reliably fall below this
                    continue
                raw_faces.append({
                    "top": y1, "right": x2, "bottom": y2, "left": x1,
                    "det_score": det_score, "embedding": face["embedding"],
                })

            # Highest det_score first, so NMS always keeps the detector's
            # own best guess for each cluster of overlapping boxes.
            raw_faces.sort(key=lambda f: f["det_score"], reverse=True)

            kept_faces = []
            for f in raw_faces:
                is_duplicate = False
                for k in kept_faces:
                    iou = _iou(f["top"], f["right"], f["bottom"], f["left"],
                               k["top"], k["right"], k["bottom"], k["left"])
                    if iou >= DETECTION_NMS_IOU:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    kept_faces.append(f)
            # ─────────────────────────────────────────────────────────────

            new_detections = []

            for f in kept_faces:
                top, right, bottom, left = f["top"], f["right"], f["bottom"], f["left"]
                emb = f["embedding"]

                with self._enc_lock:
                    name, conf = self._identify_face(emb)

                # emb travels with the detection now so a track that turns
                # out to have no spatial match can still be checked against
                # the re-ID buffer below.
                new_detections.append((top, right, bottom, left, name, conf, emb))

                if name != "Unknown":
                    if name not in self._greeted_session:
                        self._greeted_session.add(name)
                        khmer_name = self.known_khmer_names.get(name, "")
                        speak_greeting(name, khmer_name)
                        display = name.title()
                        self._recognition_queue.put({'type': 'toast', 'display': display})

                    self._queue_recognition(name, conf)

            # Merge detections into box list.
            #
            # Association strategy: GLOBAL greedy-best-first over the full
            # detection x existing-track score matrix, where score blends
            # spatial proximity (IoU + center distance) with embedding
            # similarity to the track's last known face. This replaces the
            # old per-detection-in-scan-order greedy match, which is the
            # classic cause of ID swaps: in scan order, detection #1 simply
            # took whichever track had the best LOCAL score among tracks not
            # yet used — if two people are crossing paths, detection #1
            # (say, Person B walking through Person A's old box) could grab
            # Person A's track first, forcing detection #2 (Person A) into a
            # worse leftover match or a fresh ID. Sorting all candidate
            # pairs globally and committing the single best pair first, then
            # the next best among what's left, is a strong greedy
            # approximation of the Hungarian algorithm and is symmetric:
            # whichever (detection, track) pair is the most confident match
            # in the whole frame wins regardless of scan order.
            #
            # The embedding-similarity term is what actually breaks ties
            # correctly under crowding: two crossing people can have nearly
            # identical IoU/distance scores against each other's old boxes
            # for a frame or two, but their face embeddings remain distinct,
            # so embedding similarity reliably favors the correct pairing
            # even when pure spatial overlap can't.
            with self._boxes_lock:
                existing = self._boxes.copy()

                def _embed_sim(track_id, demb):
                    prev = self._box_embeddings.get(track_id)
                    if prev is None:
                        return 0.0  # no prior embedding — spatial-only, neutral
                    return float(np.dot(prev, demb))

                candidates = []  # (score, det_idx, track_idx)
                for di, det in enumerate(new_detections):
                    dt, dr, db, dl, dname, dconf, demb = det
                    for ti, sb in enumerate(existing):
                        iou = _iou(dt, dr, db, dl, sb.tt, sb.tr, sb.tb, sb.tl)
                        cdist = _center_dist(dt, dr, db, dl,
                                             sb.tt, sb.tr, sb.tb, sb.tl)
                        esim = _embed_sim(sb.track_id, demb)

                        # KEY CHANGE from the old fixed-gate model: the
                        # acceptable search radius is now PER-TRACK, via
                        # uncertainty_radius(), which grows the longer that
                        # specific track has been blind (no detection).
                        # This is the actual fix for "identity lost when a
                        # recognized person hides their face and moves":
                        # previously every track used the same fixed
                        # max_dist*0.3 radius regardless of how long it had
                        # gone unobserved, so a track that legitimately
                        # walked 300px across the frame while blind for 2
                        # seconds could never spatially qualify as a
                        # candidate for the detection that reappears at its
                        # new, real position — it was already cdist_score=0
                        # long before reaching there. With a growing
                        # radius, a long-blind track's gate widens to match
                        # how far it could plausibly have walked.
                        radius = sb.uncertainty_radius()
                        cdist_score = max(0.0, 1.0 - cdist / max(radius, 1.0))

                        if sb.blind_since is not None:
                            # IoU is structurally unreliable for a blind
                            # track once it's moved any real distance (the
                            # predicted box and the new detection's box
                            # will not overlap even when this IS the
                            # correct match) — so for blind tracks, lean
                            # the spatial score on widened-center-distance
                            # alone, and weight embedding similarity much
                            # more heavily than for a normally-tracked
                            # detection, since appearance is now the
                            # primary evidence rather than a tie-breaker.
                            spatial = cdist_score
                            score = spatial * 0.45 + max(esim, -0.3) * 0.55
                            gate_ok = spatial > 0.15
                        else:
                            # Normal case — face/track has been detected
                            # every cycle, IoU is meaningful and primary.
                            spatial = iou * 0.7 + cdist_score * 0.3
                            score = spatial * 0.75 + max(esim, -0.3) * 0.25
                            gate_ok = spatial > 0.2

                        if gate_ok:
                            candidates.append((score, di, ti))

                candidates.sort(key=lambda c: c[0], reverse=True)

                det_assigned = {}   # det_idx -> track_idx
                track_used = set()
                for score, di, ti in candidates:
                    if di in det_assigned or ti in track_used:
                        continue
                    det_assigned[di] = ti
                    track_used.add(ti)

                matched = []
                reattached_alive_ids = set()  # existing-track indices reattached this cycle via embedding, not spatial assoc
                for di, det in enumerate(new_detections):
                    dt, dr, db, dl, dname, dconf, demb = det

                    if di in det_assigned:
                        # Spatial+embedding match — same box we already had,
                        # just move it.
                        sb = existing[det_assigned[di]]
                        sb.set_target(dt, dr, db, dl, dname, dconf)
                        self._box_embeddings[sb.track_id] = demb
                        matched.append(sb)
                        continue

                    # No spatial match this cycle. Before falling back to
                    # LostTrackBuffer (which only holds tracks that have
                    # already been fully dropped) or minting a brand new
                    # track_id, check whether this detection's embedding
                    # strongly matches a STILL-ALIVE locked track that
                    # simply didn't get a spatial candidate this round —
                    # e.g. it left frame and re-entered from a different
                    # position, or coasted via velocity prediction far from
                    # where the person actually reappeared. Without this,
                    # such a re-entry would either spawn a duplicate
                    # track_id (violates "do not create duplicate IDs") or
                    # have to wait out the full IDENTITY_PERSISTENCE_TIMEOUT
                    # before LostTrackBuffer could catch it — this closes
                    # that gap and reattaches immediately.
                    best_alive_ti, best_alive_sim = -1, -1.0
                    for ti, sb in enumerate(existing):
                        if ti in track_used or ti in reattached_alive_ids:
                            continue
                        if not sb.locked:
                            continue  # only locked identities get this benefit of the doubt
                        prev_emb = self._box_embeddings.get(sb.track_id)
                        if prev_emb is None:
                            continue
                        sim = float(np.dot(prev_emb, demb))
                        if sim > best_alive_sim:
                            best_alive_sim, best_alive_ti = sim, ti

                    if best_alive_ti >= 0 and best_alive_sim >= REID_SIMILARITY_THRESHOLD:
                        sb = existing[best_alive_ti]
                        log.debug(
                            f"Track {sb.track_id} ('{sb.name}') reattached "
                            f"via embedding after losing spatial contact "
                            f"(sim={best_alive_sim:.3f})")
                        sb.set_target(dt, dr, db, dl, dname, dconf)
                        self._box_embeddings[sb.track_id] = demb
                        matched.append(sb)
                        track_used.add(best_alive_ti)
                        reattached_alive_ids.add(best_alive_ti)
                        continue

                    # Still nothing — before assuming this is a brand new
                    # face, check whether it's someone who was lost
                    # recently (occlusion, turned away, walked off and
                    # back, and already fully timed out of self._boxes).
                    # If so, resume their track_id rather than starting a
                    # fresh one.
                    reid_id, reid_name, reid_conf = \
                        self._lost_tracks.match(demb)

                    if reid_id is not None:
                        log.debug(
                            f"Re-ID: '{reid_name}' resumed track {reid_id} "
                            f"after occlusion/absence")
                        sb = SmoothBox(dt, dr, db, dl, reid_name, reid_conf,
                                       track_id=reid_id)
                    else:
                        sb = SmoothBox(dt, dr, db, dl, dname, dconf)

                    self._box_embeddings[sb.track_id] = demb
                    matched.append(sb)

                # Boxes that weren't matched to any detection this cycle.
                #
                # Two independent questions here, deliberately NOT conflated:
                #   1. Should this box keep COASTING in self._boxes (visual
                #      prediction, may still render if not yet fully faded)?
                #      -> governed by is_stale() / FADE_OUT_TIME, short.
                #   2. Should the IDENTITY itself be considered lost and
                #      handed off to the Re-ID buffer?
                #      -> governed by identity_expired(), which uses the
                #      much longer IDENTITY_PERSISTENCE_TIMEOUT for locked
                #      tracks. A locked track can visually fade out (alpha
                #      drops to 0, nothing drawn) while STILL being kept
                #      alive in self._boxes for up to IDENTITY_PERSISTENCE_
                #      TIMEOUT seconds, ready to be instantly reattached the
                #      moment a spatial detection reappears anywhere near
                #      its predicted/last position — no Re-ID embedding
                #      lookup needed, no re-greet, no streak to rebuild,
                #      because it's literally still the same live track
                #      object. This is what prevents Name -> Unknown ->
                #      Name flicker during a real but brief occlusion: the
                #      track never actually left self._boxes in the first
                #      place during a normal-length disappearance.
                for ti, sb in enumerate(existing):
                    if ti in track_used:
                        continue

                    # No detection matched this track this cycle — start
                    # (or continue) the blind-tracking clock. This is what
                    # uncertainty_radius() and lerp_dt()'s slower
                    # BLIND_VELOCITY_DAMPING key off of, so a track that's
                    # had its face hidden for several cycles in a row
                    # progressively widens its acceptable re-association
                    # area instead of using the same fixed gate every time.
                    if sb.blind_since is None:
                        sb.blind_since = time.time()

                    if sb.identity_expired():
                        # Genuinely gone long enough (per its own
                        # locked/unlocked timeout) — save its last known
                        # embedding so a re-appearance within
                        # REID_TTL_SECONDS can still resume this same
                        # track_id via embedding match, then drop it.
                        emb = self._box_embeddings.pop(sb.track_id, None)
                        self._lost_tracks.save(sb, emb)
                    else:
                        # Identity still within its grace period — keep the
                        # track alive regardless of visual staleness. It
                        # will still coast on velocity prediction and
                        # render (or not, once fully visually faded) via
                        # the normal staleness()/is_stale() path in
                        # _render_loop; that's a purely cosmetic concern
                        # separate from whether we still "believe" this is
                        # the same person.
                        matched.append(sb)

                # ── Stage 2 safety net: same-person track merge ───────────
                # Stage 1 NMS (above) prevents duplicate RAW DETECTIONS from
                # ever reaching the tracker in the first place, which is the
                # actual root cause fix. This second pass is a belt-and-
                # suspenders guard against the SEPARATE failure mode of two
                # already-EXISTING tracks both representing one physical
                # person — e.g. a duplicate minted before this fix existed
                # in a long-running session, or a brand-new track spawned
                # this very cycle (the "no spatial/embedding match, no
                # Re-ID match" branch above) that happens to land directly
                # on top of an already-tracked person because the detector
                # produced a same-cycle box NMS didn't catch (different
                # detect_score ordering edge case, or a box that fell just
                # under DETECTION_NMS_IOU but is still clearly the same
                # face by position).
                #
                # Two boxes in `matched` are treated as the same physical
                # person if they overlap heavily (IoU) AND are not both
                # confidently-different known identities (two real known
                # people standing close are NOT merged — only merged when
                # at least one side is Unknown, or both sides agree). When
                # merged, the higher-confidence / known-over-Unknown box is
                # kept and the other is dropped — "never display both a
                # recognized name and Unknown for the same person".
                TRACK_MERGE_IOU = 0.55

                def _quality(sb):
                    # Sort key: prefer locked, then known-over-Unknown, then
                    # higher confidence, then the OLDER track (smaller
                    # track_id = created earlier = more accumulated history).
                    return (sb.locked, sb.name != "Unknown", sb.confidence, -sb.track_id)

                dropped_ids = set()
                for i, sb_a in enumerate(matched):
                    if sb_a.track_id in dropped_ids:
                        continue
                    for sb_b in matched[i + 1:]:
                        if sb_b.track_id in dropped_ids:
                            continue
                        iou = _iou(sb_a.dt, sb_a.dr, sb_a.db, sb_a.dl,
                                   sb_b.dt, sb_b.dr, sb_b.db, sb_b.dl)
                        if iou < TRACK_MERGE_IOU:
                            continue
                        if sb_a.name != "Unknown" and sb_b.name != "Unknown" \
                                and sb_a.name != sb_b.name:
                            # Two DIFFERENT confidently-known people heavily
                            # overlapping (e.g. standing shoulder to
                            # shoulder) — a real scenario, not a duplicate.
                            # Leave both; do not merge.
                            continue
                        # Same person (one Unknown + one known, both
                        # Unknown, or both the same known name) — keep
                        # whichever track is better-established and drop
                        # the other. If sb_a itself ends up being the one
                        # dropped, stop comparing it against further boxes
                        # this pass (it's gone) and move on to the next i.
                        if _quality(sb_a) >= _quality(sb_b):
                            keep, drop = sb_a, sb_b
                        else:
                            keep, drop = sb_b, sb_a
                        log.debug(
                            f"Merging duplicate track {drop.track_id} "
                            f"('{drop.name}') into {keep.track_id} "
                            f"('{keep.name}') — IoU={iou:.2f}")
                        dropped_ids.add(drop.track_id)
                        self._box_embeddings.pop(drop.track_id, None)
                        if drop.track_id == sb_a.track_id:
                            break  # sb_a no longer exists; stop its inner loop

                matched = [sb for sb in matched if sb.track_id not in dropped_ids]
                # ───────────────────────────────────────────────────────────

                self._boxes = matched

    def _queue_recognition(self, name, confidence):
        if name in self.today_logged:
            return
        now = time.time()
        self._pending_recognition = {
            k: v for k, v in self._pending_recognition.items()
            if now - v['last_seen'] < 3.0}
        if name not in self._pending_recognition:
            self._pending_recognition[name] = {
                'count': 1, 'last_seen': now, 'confidence': confidence}
        else:
            self._pending_recognition[name].update({
                'count': self._pending_recognition[name]['count'] + 1,
                'last_seen': now,
                'confidence': confidence,
            })
        if self._pending_recognition[name]['count'] >= 3:
            self.today_logged.add(name)
            payload = dict(self._pending_recognition[name])
            payload['name'] = name
            payload['date'] = str(date.today())
            self._recognition_queue.put(payload)
            del self._pending_recognition[name]

    def _process_recognition_queue(self):
        try:
            while True:
                payload = self._recognition_queue.get_nowait()
                if payload.get('type') == 'toast':
                    self.show_greeting_toast(payload['display'])
                    continue
                now = datetime.now().strftime("%H:%M:%S")
                database.log_recognition(
                    payload['name'], payload['date'], now,
                    confidence=payload.get('confidence', 0.0),
                )
                self._append_live_log(f"[{now}]  {payload['name']}\n", payload['date'])
                log.info(f"Recognition logged: {payload['name']} at {now}")
        except queue.Empty:
            pass
        finally:
            self._process_id = self.after(100, self._process_recognition_queue)

    # ── Face Recognition ──────────────────────────────────────────────────────

    def _identify_face(self, embedding: np.ndarray):
        """
        Identify a face embedding against the known gallery.

        Two gates must BOTH pass before a name is returned instead of
        "Unknown":
          1. Absolute gate — best similarity >= CONFIDENCE_THRESHOLD.
          2. Margin gate    — the best name's score must beat the best score
             belonging to any OTHER enrolled name by at least IDENTITY_MARGIN.

        Gate 2 is what actually prevents confusing two different enrolled
        people: gate 1 alone only proves "this looks enough like Person A's
        photos", it says nothing about whether it looks just as much like
        Person B's. A close, ambiguous embedding (similar-looking siblings,
        bad angle/lighting) can clear an absolute threshold for two
        different identities at once; the margin gate refuses to pick a
        winner in that case and reports Unknown instead, per "never guess —
        prefer a miss over a misidentification".

        NOTE: known_names/known_encodings hold one row PER ENROLLED PHOTO,
        so the same person can legitimately occupy several rows (multi-photo
        enrollment). The margin must therefore be computed as
        best-score-for-the-winning-name vs best-score-for-any-OTHER-name,
        not "row 1 vs row 2" — otherwise a person with multiple good photos
        would be penalized against their own other photos.
        """
        if self._known_matrix is None or self._known_matrix.shape[0] == 0:
            return "Unknown", 0.0

        sims = self._known_matrix @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_name = self.known_names[best_idx]

        if best_sim < CONFIDENCE_THRESHOLD:
            return "Unknown", best_sim

        # Best score among rows belonging to a DIFFERENT name than the winner.
        other_mask = np.array(
            [n != best_name for n in self.known_names], dtype=bool)
        if other_mask.any():
            second_sim = float(np.max(sims[other_mask]))
        else:
            second_sim = -1.0  # only one identity enrolled at all — no rival

        if (best_sim - second_sim) >= IDENTITY_MARGIN:
            return best_name, best_sim

        # Too close to a rival identity to safely disambiguate — report
        # Unknown rather than risk a false identification.
        return "Unknown", best_sim

    def _append_live_log(self, entry, today):
        """Append text to the QTextEdit live log (called from main thread via after())."""
        try:
            self.live_log.append(entry.rstrip())
            # Scroll to bottom
            sb = self.live_log.verticalScrollBar()
            sb.setValue(sb.maximum())
            count = len(self.today_logged)
            self.stat_today.setText(str(count))
            self.live_log_count.setText(f"{count} seen today")
        except Exception:
            pass

    # ── Registration (multi-photo) ────────────────────────────────────────────

