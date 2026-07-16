"""
core.tracking
=============
Per-face tracking state and logic: SmoothBox (velocity-predicted,
identity-locking bounding box), LostTrackBuffer (short-term embedding
memory for re-ID across occlusion), IoU/center-distance helpers, and the
on-screen tracer/scan-line drawing helpers.
"""

import threading
import time
import collections

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

import logging
log = logging.getLogger("FaceRecog")

from .config import (
    CONFIDENCE_THRESHOLD, IDENTITY_MARGIN, IDENTITY_CONFIRM_FRAMES,
    IDENTITY_SWAP_CONFIRM_FRAMES, IDENTITY_SWAP_MARGIN, IDENTITY_LOCK_STREAK,
    IDENTITY_LOCK_MIN_AVG_CONF, CONF_SMOOTHING_WINDOW, ASSOC_EMBED_VETO_SIM,
    BLIND_VELOCITY_DAMPING, BLIND_UNCERTAINTY_GROWTH_PX, BLIND_UNCERTAINTY_MAX_PX,
    DETECT_SCALE, DETECT_EVERY_N_FRAMES, DISPLAY_WIDTH, DISPLAY_HEIGHT,
    LERP_SPEED_MIN, LERP_SPEED_MAX, VELOCITY_DAMPING, PREDICTION_WEIGHT,
    FADE_OUT_TIME, IDENTITY_PERSISTENCE_TIMEOUT, SCAN_LINE_SPEED,
    UNKNOWN_STREAK_TO_DROP, REID_TTL_SECONDS, REID_SIMILARITY_THRESHOLD,
    REID_MARGIN,
)

# ── SmoothBox ─────────────────────────────────────────────────────────────────
_next_track_id = 0
_track_id_lock = threading.Lock()


def _new_track_id() -> int:
    """Monotonic unique id for each tracked face — used to key the re-ID
    (LostTrackBuffer) dict and to dedupe greetings across an occlusion."""
    global _next_track_id
    with _track_id_lock:
        _next_track_id += 1
        return _next_track_id

class SmoothBox:
    __slots__ = (
        'name', 'confidence', 'track_id',
        'dt', 'dr', 'db', 'dl',
        'tt', 'tr', 'tb', 'tl',
        'vt', 'vr', 'vb', 'vl',
        'last_update', 'birth_time',
        'last_detect_t',
        'pending_name', 'pending_conf', 'pending_streak',
        'confirm_streak', 'locked',
        'conf_history',
        'last_identity_evidence',
        'blind_since',
    )

    def __init__(self, top, right, bottom, left, name, confidence, track_id=None):
        self.name = name
        self.confidence = confidence
        self.track_id = track_id if track_id is not None else _new_track_id()
        self.dt = float(top);    self.dr = float(right)
        self.db = float(bottom); self.dl = float(left)
        self.tt = float(top);    self.tr = float(right)
        self.tb = float(bottom); self.tl = float(left)
        self.vt = 0.0; self.vr = 0.0; self.vb = 0.0; self.vl = 0.0
        self.last_update = time.time()
        self.birth_time  = time.time()
        self.last_detect_t = time.time()
        # Identity-change confirmation: a candidate new identity must be
        # seen N times in a row (consecutive detect cycles, no disagreement
        # in between) before it actually overwrites self.name. This is what
        # stops a single noisy/borderline frame from flipping who a track
        # is displayed as — see set_target(). N is IDENTITY_CONFIRM_FRAMES
        # for an Unknown track, or the stricter IDENTITY_SWAP_CONFIRM_FRAMES
        # once the track is locked (see below).
        self.pending_name   = None
        self.pending_conf   = 0.0
        self.pending_streak = 0
        # How many consecutive detect cycles in a row have agreed with the
        # CURRENT self.name (reset to 0 the instant a different candidate
        # starts accumulating in pending_*). Used to decide when a track
        # graduates from "freshly assigned" to "locked" — see `locked`.
        self.confirm_streak = 1 if name != "Unknown" else 0
        # Once True, this track is treated as a confidently, durably
        # recognized identity: it gets the long IDENTITY_PERSISTENCE_TIMEOUT
        # grace period during occlusion/absence, and can only be swapped to
        # a different known identity under the stricter
        # IDENTITY_SWAP_CONFIRM_FRAMES / IDENTITY_SWAP_MARGIN rules instead
        # of the lighter initial-assignment rules. A track is never
        # unlocked once locked (within a session) — that's the whole point
        # of "lock onto identity once confirmed".
        self.locked = False
        # Rolling window of recent per-frame confidence samples (only
        # appended on frames that AGREE with the current self.name) used to
        # compute an averaged confidence instead of trusting whatever the
        # single latest frame happened to score. This is what the lock
        # decision (IDENTITY_LOCK_MIN_AVG_CONF) checks against, and it's
        # also what's reported as sb.confidence so the on-screen % reflects
        # sustained accuracy rather than one lucky/unlucky frame.
        self.conf_history = collections.deque(maxlen=CONF_SMOOTHING_WINDOW)
        if name != "Unknown":
            self.conf_history.append(confidence)
        # Wall-clock time this track started going "blind" (no detection
        # matched to it this cycle), or None while it's currently being
        # detected normally. This is what drives the growing-uncertainty
        # motion model: the search radius for re-association widens the
        # longer blind_since has been set, instead of using a fixed gate
        # regardless of how long the person has been faceless. Reset to
        # None the instant any detection (even an Unknown one — body still
        # visible, face just not confidently recognized) reattaches.
        self.blind_since = None
        # Separate from last_update (which only governs the on-screen
        # fade/coast and box removal). This timestamp only advances when
        # there is POSITIVE evidence for the current identity — either a
        # detection that agreed with self.name, or (for an Unknown track)
        # any detection at all. It does NOT advance on a transient "Unknown"
        # reading from an already-identified track, and does NOT advance
        # just because the box is coasting on velocity prediction with no
        # real detection. This is the clock IDENTITY_PERSISTENCE_TIMEOUT
        # checks — "how long since we last had real evidence this is still
        # the same confirmed person" — independent of how long the box has
        # been visually fading.
        self.last_identity_evidence = time.time()

    def set_target(self, top, right, bottom, left, name, confidence):
        now = time.time()
        elapsed = max(now - self.last_detect_t, 1e-3)
        # After a long blind gap, the raw position delta divided by the
        # full gap duration would produce a heavily diluted/misleading
        # velocity estimate (e.g. "they moved 300px over 4 blind seconds"
        # => 75px/s, even if they actually arrived in the last 0.2s and
        # were stationary before that). Cap the elapsed time used for THIS
        # single velocity recomputation so a fresh re-detection establishes
        # a clean new velocity baseline from approximately this frame
        # interval, rather than inheriting a smeared average over the
        # entire blind period.
        if self.blind_since is not None:
            elapsed = min(elapsed, 1.0 / max(DETECT_EVERY_N_FRAMES, 1) + 0.2)
        raw_vt = (float(top)    - self.tt) / elapsed
        raw_vr = (float(right)  - self.tr) / elapsed
        raw_vb = (float(bottom) - self.tb) / elapsed
        raw_vl = (float(left)   - self.tl) / elapsed
        self.vt = raw_vt * VELOCITY_DAMPING
        self.vr = raw_vr * VELOCITY_DAMPING
        self.vb = raw_vb * VELOCITY_DAMPING
        self.vl = raw_vl * VELOCITY_DAMPING
        self.tt, self.tr = float(top), float(right)
        self.tb, self.tl = float(bottom), float(left)
        # Any real detection — known, Unknown, or re-attached after a
        # blind gap — is direct spatial evidence the track has been
        # re-located. Clear blind_since unconditionally here so
        # uncertainty_radius() snaps back to the tight base radius; the
        # widened search was only meant to apply WHILE no detection exists
        # to correct the prediction.
        self.blind_since = None

        # Permanent identity lock: once this track has been identified as a
        # known person, "Unknown" can NEVER overwrite that name — not from
        # turning away, motion blur, partial occlusion, low confidence,
        # nothing. Only a DIFFERENT known name can replace a known name,
        # and even then only after sustained agreement (see below). Unknown
        # tracks can still become known when recognized.
        #
        # Critically: this branch is still spatial/tracking evidence that
        # the SAME physical track is still being followed (the detector
        # found a face here, even if recognition itself came back
        # unconfident) — that alone is enough to refresh
        # last_identity_evidence and keep the persistence clock from
        # expiring, even though it can't refresh conf_history or override
        # the name itself.
        if self.name != "Unknown" and name == "Unknown":
            self.pending_name = None
            self.pending_streak = 0
            self.last_update = now
            self.last_detect_t = now
            self.last_identity_evidence = now
            return

        if name == self.name:
            # Detection agrees with current identity — nothing to confirm.
            # Feed the smoothing window and report the AVERAGE as
            # self.confidence, not the raw single-frame value, so the
            # displayed/logged confidence reflects sustained recognition
            # quality rather than one noisy sample.
            if name != "Unknown":
                self.conf_history.append(confidence)
                self.confidence = sum(self.conf_history) / len(self.conf_history)
                self.confirm_streak += 1
                if (not self.locked
                        and self.confirm_streak >= IDENTITY_LOCK_STREAK
                        and self.confidence >= IDENTITY_LOCK_MIN_AVG_CONF):
                    self.locked = True
            else:
                self.confidence = confidence
            self.pending_name = None
            self.pending_streak = 0
            self.last_update = now
            self.last_detect_t = now
            self.last_identity_evidence = now
            return

        # `name` differs from the track's current identity (either the
        # track is currently "Unknown" and this is a candidate known
        # identity, or it's currently a known person and a DIFFERENT known
        # name was detected). Require several consecutive agreeing
        # detections on this same candidate before committing it — any
        # disagreement in between resets the streak, so noisy single-frame
        # misreads can never accumulate piecemeal into a swap.
        #
        # A LOCKED track demands a much higher bar to be overridden
        # (IDENTITY_SWAP_CONFIRM_FRAMES consecutive agreeing frames, all of
        # which must individually clear IDENTITY_SWAP_MARGIN over their
        # own runner-up — enforced upstream in _identify_face's normal
        # margin check, this just adds frame-count pressure) than a fresh
        # Unknown track does (IDENTITY_CONFIRM_FRAMES) — a confidently
        # locked identity should be very hard to dislodge by anything
        # short of overwhelming, sustained evidence that it's really a
        # different person (e.g. the original person left and someone
        # else is now in frame).
        required_streak = IDENTITY_SWAP_CONFIRM_FRAMES if self.locked else IDENTITY_CONFIRM_FRAMES

        if self.pending_name == name:
            self.pending_streak += 1
            self.pending_conf = confidence
        else:
            self.pending_name = name
            self.pending_conf = confidence
            self.pending_streak = 1

        if self.pending_streak >= required_streak:
            log.debug(
                f"Track {self.track_id} identity swap: "
                f"'{self.name}' -> '{self.pending_name}' "
                f"(was_locked={self.locked}, streak={self.pending_streak})")
            self.name = self.pending_name
            self.confidence = self.pending_conf
            self.pending_name = None
            self.pending_streak = 0
            self.confirm_streak = 1
            self.locked = False  # must re-earn lock under the new identity
            self.conf_history.clear()
            if self.name != "Unknown":
                self.conf_history.append(self.confidence)

        self.last_update = now
        self.last_detect_t = now
        self.last_identity_evidence = now

    def lerp_dt(self, dt: float):
        frame_alpha_min = 1.0 - (1.0 - LERP_SPEED_MIN) ** (dt * 60.0)
        frame_alpha_max = 1.0 - (1.0 - LERP_SPEED_MAX) ** (dt * 60.0)
        self.tt += self.vt * dt
        self.tr += self.vr * dt
        self.tb += self.vb * dt
        self.tl += self.vl * dt
        pred_tt = self.tt + self.vt * PREDICTION_WEIGHT
        pred_tr = self.tr + self.vr * PREDICTION_WEIGHT
        pred_tb = self.tb + self.vb * PREDICTION_WEIGHT
        pred_tl = self.tl + self.vl * PREDICTION_WEIGHT
        dist = abs(pred_tt - self.dt) + abs(pred_tr - self.dr) + \
               abs(pred_tb - self.db) + abs(pred_tl - self.dl)
        norm = min(dist / 100.0, 1.0)
        a = frame_alpha_min + (frame_alpha_max - frame_alpha_min) * norm
        self.dt += (pred_tt - self.dt) * a
        self.dr += (pred_tr - self.dr) * a
        self.db += (pred_tb - self.db) * a
        self.dl += (pred_tl - self.dl) * a
        # While the track has a real detection every cycle, decay velocity
        # quickly toward zero (VELOCITY_DAMPING) — appropriate because any
        # actual movement is being directly re-measured and re-applied via
        # set_target() anyway, so aggressive damping here just keeps the
        # box from overshooting between detections.
        #
        # While BLIND (no detection this track_id has matched in a while),
        # damping must be much slower (BLIND_VELOCITY_DAMPING): a person
        # who was walking with their face hidden does not stop walking
        # just because the detector lost their face. Keeping the velocity
        # alive and projecting it forward is what lets the predicted
        # position track roughly where they're actually walking TO, rather
        # than decaying back to where they were last SEEN — which is
        # exactly the old bug: aggressive damping meant a blind, moving
        # track's predicted box would sit near its last-seen position while
        # the real person kept walking away from it, so by the time their
        # face reappeared the predicted and real positions had diverged far
        # past any fixed spatial association gate.
        damp_rate = BLIND_VELOCITY_DAMPING if self.blind_since is not None else VELOCITY_DAMPING
        damping = damp_rate ** (dt * 60.0)
        self.vt *= damping; self.vr *= damping
        self.vb *= damping; self.vl *= damping

    def uncertainty_radius(self) -> float:
        """
        How far (in pixels) this track's predicted position is allowed to
        have drifted from where a new detection actually is, and still be
        considered a plausible re-association candidate.

        This is the crux of the redesign: instead of a single fixed gate
        used for every track regardless of its situation, the radius GROWS
        the longer the track has been blind (no detection matched it),
        replicating how a Kalman filter's positional covariance grows
        during predict-only steps with no observation to correct it. A
        track that's been tracking normally every cycle gets a tight
        radius (current bbox size — faces don't teleport between 30fps
        frames). A track that's been blind for 2 seconds while the person
        was walking gets a much wider radius, because we genuinely don't
        know their exact position anymore, only roughly how far they
        could plausibly have moved.
        """
        base = max(self.tr - self.tl, self.tb - self.tt, 20.0)
        if self.blind_since is None:
            return base
        blind_elapsed = time.time() - self.blind_since
        speed = (self.vt ** 2 + self.vl ** 2) ** 0.5  # rough px/sec magnitude from top/left velocity
        growth = min(blind_elapsed * (BLIND_UNCERTAINTY_GROWTH_PX + speed * 0.5),
                     BLIND_UNCERTAINTY_MAX_PX)
        return base + growth

    def lerp(self):
        self.lerp_dt(1.0 / 60.0)

    def ints(self):
        return int(self.dt), int(self.dr), int(self.db), int(self.dl)

    def staleness(self):
        """Purely visual fade fraction — governs on-screen opacity/coasting.
        Independent of identity persistence; see identity_expired()."""
        return min((time.time() - self.last_update) / FADE_OUT_TIME, 1.0)

    def is_stale(self, timeout=None):
        """Whether the box should stop being drawn / be dropped from the
        render list. Purely visual — does NOT mean the identity itself
        should be forgotten; see identity_expired() / LostTrackBuffer."""
        timeout = timeout or FADE_OUT_TIME
        return time.time() - self.last_update > timeout

    def identity_expired(self):
        """
        Whether this track has gone long enough with NO positive identity
        evidence (no detection at all, or no detection agreeing with its
        current name) that the identity itself should be considered lost —
        as opposed to merely the box being visually stale.

        Locked tracks get the full IDENTITY_PERSISTENCE_TIMEOUT grace
        period (default 8s, configurable 3-10s) — this is what lets a
        confirmed identity survive a head turn, full occlusion, brief
        exit-and-reentry, blur, or lighting change without flickering to
        Unknown. Tracks that were never locked (e.g. still Unknown, or
        only seen for a couple of frames before disappearing) use the
        shorter FADE_OUT_TIME instead — they haven't earned the extended
        benefit of the doubt yet.
        """
        timeout = IDENTITY_PERSISTENCE_TIMEOUT if self.locked else FADE_OUT_TIME
        return time.time() - self.last_identity_evidence > timeout

    def center(self):
        return (self.dl + self.dr) / 2.0, (self.dt + self.db) / 2.0


def _iou(at, ar, ab, al, bt, br, bb, bl):
    it = max(at, bt); ib = min(ab, bb)
    il = max(al, bl); ir = min(ar, br)
    inter = max(0, ib - it) * max(0, ir - il)
    if inter == 0:
        return 0.0
    area1 = (ab - at) * (ar - al)
    area2 = (bb - bt) * (br - bl)
    return inter / (area1 + area2 - inter)


def _center_dist(at, ar, ab, al, bt, br, bb, bl):
    cx1, cy1 = (al + ar) / 2.0, (at + ab) / 2.0
    cx2, cy2 = (bl + br) / 2.0, (bt + bb) / 2.0
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

# ── LostTrackBuffer (re-ID across occlusion / walk-away) ────────────────────
class LostTrackBuffer:
    """
    Short-term memory of recently-lost tracks, keyed by track_id.

    SmoothBox already carries a face forward via velocity prediction while
    it's merely undetected for a frame or two (motion blur, a missed
    detect cycle). But once a box has been stale for FADE_OUT_TIME and is
    about to be dropped from self._boxes entirely, its identity embedding
    is saved here instead. If a new, spatially-unrelated detection shows up
    within REID_TTL_SECONDS and its embedding is close enough (cosine sim
    >= REID_SIMILARITY_THRESHOLD) to a saved entry, that's almost certainly
    the same person returning — e.g. they turned away from the camera,
    walked behind another person/object, or stepped briefly out of frame —
    so we hand back their original track_id instead of minting a new one.
    This keeps the greeting-once-per-session logic correct and avoids
    spurious duplicate entries in recognition history for one continuous
    visit.
    """

    def __init__(self):
        self._tracks: dict[int, dict] = {}
        self._lock = threading.Lock()

    def save(self, sb: 'SmoothBox', embedding):
        """Stash a box's identity right before it gets dropped as stale."""
        if embedding is None or sb.name == "Unknown":
            return
        with self._lock:
            self._tracks[sb.track_id] = {
                'embedding':  embedding.copy(),
                'name':       sb.name,
                'confidence': sb.confidence,
                'saved_at':   time.time(),
            }

    def match(self, query_emb):
        """
        Look for a saved track whose embedding matches query_emb closely
        enough. Expired entries (older than REID_TTL_SECONDS) are purged
        first. Returns (track_id, name, confidence) or (None, '', 0.0).
        A match is consumed (removed) so it can't be re-claimed twice.
        """
        now = time.time()
        with self._lock:
            expired = [tid for tid, v in self._tracks.items()
                       if now - v['saved_at'] > REID_TTL_SECONDS]
            for tid in expired:
                del self._tracks[tid]

            if not self._tracks:
                return None, '', 0.0

            tids = list(self._tracks.keys())
            embs = np.stack([self._tracks[t]['embedding'] for t in tids])
            sims = embs @ query_emb
            order = np.argsort(sims)[::-1]
            best = int(order[0])
            best_sim = float(sims[best])
            second_sim = float(sims[order[1]]) if len(order) > 1 else -1.0

            if best_sim >= REID_SIMILARITY_THRESHOLD and \
                    (best_sim - second_sim) >= REID_MARGIN:
                rec = self._tracks.pop(tids[best])
                return tids[best], rec['name'], rec['confidence']

        return None, '', 0.0


def _draw_corner_tracers(frame, left, top, right, bottom, color,
                          thickness=2, corner_len=20, alpha=1.0):
    cl = max(corner_len, int((right - left) * 0.2))
    cl = min(cl, (right - left) // 2, (bottom - top) // 2)
    overlay = frame.copy() if alpha < 1.0 else frame
    cv2.line(overlay, (left, top),     (left + cl, top),     color, thickness)
    cv2.line(overlay, (left, top),     (left, top + cl),     color, thickness)
    cv2.line(overlay, (right, top),    (right - cl, top),    color, thickness)
    cv2.line(overlay, (right, top),    (right, top + cl),    color, thickness)
    cv2.line(overlay, (left, bottom),  (left + cl, bottom),  color, thickness)
    cv2.line(overlay, (left, bottom),  (left, bottom - cl),  color, thickness)
    cv2.line(overlay, (right, bottom), (right - cl, bottom), color, thickness)
    cv2.line(overlay, (right, bottom), (right, bottom - cl), color, thickness)
    thin_color = tuple(int(c * 0.35) for c in color)
    cv2.rectangle(overlay, (left, top), (right, bottom), thin_color, 1)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_scan_line(frame, left, top, right, bottom, color, thickness=2):
    t = time.time() * SCAN_LINE_SPEED
    frac = (np.sin(t) + 1.0) / 2.0
    y = int(top + (bottom - top) * frac)
    fade_color = tuple(int(c * 0.55) for c in color)
    cv2.line(frame, (left + 2, y), (right - 2, y), fade_color, thickness)


