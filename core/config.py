"""
core.config
===========
All tunable constants for the face-recognition pipeline: identity
confirmation/locking thresholds, blind-tracking motion model, display
sizing, and re-ID (occlusion recovery) settings.

Split out of the original main.py so every module that needs a threshold
imports it from one place instead of relying on module-level globals
defined inside the app's entry point.
"""

import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

CONFIDENCE_THRESHOLD  = 0.50
# Margin gate: best match must beat the *second*-best enrolled identity by
# this much cosine-similarity, not just clear the absolute threshold. This
# is what actually stops two visually-similar people from being confused —
# a single absolute threshold only asks "is this close enough to Person A?"
# and says nothing about "...and not equally close to Person B?". Without a
# margin check, two lookalikes (or one person's bad-angle embedding) can sit
# right next to each other in embedding space and flip identity frame to
# frame depending on noise.
IDENTITY_MARGIN          = 0.08
# A track's *displayed* name only changes from Unknown -> a known identity
# after this many consecutive detect cycles agree on the candidate. This
# converts a single lucky/unlucky frame from being able to assign an
# identity into requiring sustained agreement first.
IDENTITY_CONFIRM_FRAMES  = 3
# Once a track is LOCKED (see IDENTITY_LOCK_STREAK below), overriding it
# with a DIFFERENT known identity requires more sustained agreement than
# the initial Unknown->known assignment, and a wider margin. Swapping a
# confirmed, trusted identity is a worse failure than slowly granting a
# new one, so the bar to do it must be higher, not equal.
IDENTITY_SWAP_CONFIRM_FRAMES = 8
IDENTITY_SWAP_MARGIN          = 0.15
# How many consecutive confirmed-agreeing detect cycles before a track is
# considered "locked" — i.e. trusted enough that the stricter swap rules
# above apply instead of the lighter initial-assignment rules. Locking
# additionally requires the AVERAGED confidence (see CONF_SMOOTHING_WINDOW)
# to clear IDENTITY_LOCK_MIN_AVG_CONF, not just a string of bare-threshold
# passes — a person who barely keeps clearing CONFIDENCE_THRESHOLD frame
# after frame is not "confidently recognized" in the way a person who
# scores consistently high is.
IDENTITY_LOCK_STREAK         = 5
IDENTITY_LOCK_MIN_AVG_CONF   = 0.58
# How many recent per-frame confidence samples are kept and averaged for
# a track's displayed/logged confidence and for the lock decision above.
# Temporal averaging instead of last-sample-wins smooths out the natural
# frame-to-frame noise from pose/lighting/blur without needing any single
# frame to be perfect.
CONF_SMOOTHING_WINDOW        = 10
# Track association uses embedding similarity as a tie-breaker / veto when
# multiple existing boxes are spatially plausible matches for one detection
# (e.g. two people crossing paths) — prevents ID swaps under crowding.
ASSOC_EMBED_VETO_SIM     = 0.35

# ── Blind-tracking motion model ──────────────────────────────────────────────
# This is the core fix for "identity lost when a recognized person hides
# their face AND moves". The old model treated every undetected frame
# identically regardless of how long the track had been blind: velocity
# decayed toward zero every tick (VELOCITY_DAMPING) and the spatial
# association gate used a FIXED search radius. That combination is only
# correct for a person who is roughly stationary while hidden — the
# instant they walk while hidden, the predicted position drifts away from
# their true position and the fixed-radius gate can no longer find them
# when their face reappears, so they either become a new track or Unknown.
#
# A real tracker (Kalman filter) keeps predicting along the last known
# velocity and grows its POSITIONAL UNCERTAINTY the longer it goes without
# a fresh observation — so the longer the gap, the WIDER the area it's
# willing to accept as "could plausibly be this same track" on the next
# observation. We replicate that core idea with an explicit scalar
# uncertainty radius rather than a full covariance matrix (the codebase's
# existing lerp-based renderer doesn't need full Kalman state, just a
# correct search-gate policy), which is both simpler to integrate here and
# auditable.
BLIND_VELOCITY_DAMPING       = 0.97  # while undetected, velocity decays much more slowly than VELOCITY_DAMPING (0.70) — a walking person doesn't stop just because their face is hidden
BLIND_UNCERTAINTY_GROWTH_PX  = 140.0  # pixels/second the acceptable search radius grows per second blind, on top of pure velocity-based displacement — accounts for unpredictable turns/speed changes a constant-velocity model can't capture
BLIND_UNCERTAINTY_MAX_PX     = 480.0  # cap so a track blind for a very long time doesn't end up able to claim literally anything in frame
DETECT_SCALE          = 0.50
DETECT_EVERY_N_FRAMES = 1
DISPLAY_WIDTH         = 960
DISPLAY_HEIGHT        = 540
LERP_SPEED_MIN        = 0.45
LERP_SPEED_MAX        = 0.97
VELOCITY_DAMPING      = 0.70
PREDICTION_WEIGHT     = 0.015   # seconds of velocity look-ahead
# Purely visual: how long a box keeps fading out / coasting on velocity
# prediction on screen after its last detection match, before it's removed
# from the render list. This is now ONLY about drawing — it no longer
# controls how long an IDENTITY is trusted; see IDENTITY_PERSISTENCE_TIMEOUT
# below for that. Kept short so the on-screen box doesn't visually drift far
# from where the person actually is once tracking has nothing to go on.
FADE_OUT_TIME         = 5.0
# How long (seconds) a LOCKED identity is retained and kept displayed even
# while the face cannot currently be matched/detected at all — covers
# turning away, being fully occluded, leaving frame briefly, motion blur,
# bad lighting, etc. This is intentionally longer than FADE_OUT_TIME: the
# box itself may visually fade/coast on prediction, but the identity BINDING
# to that track survives independently for up to this long. Configurable
# 3-10s per the brief; 8s default balances "don't flicker to Unknown during
# a normal occlusion" against "don't keep showing a name for someone who
# may genuinely be gone". Only locked tracks (see IDENTITY_LOCK_STREAK) get
# this extended grace period — a track that was only ever seen for 1-2
# frames and never locked doesn't get the same benefit of the doubt.
IDENTITY_PERSISTENCE_TIMEOUT = 8.0
SCAN_LINE_SPEED       = 3.0
# How many consecutive "Unknown" detections a track must accumulate before
# its displayed name is allowed to drop from a known identity to Unknown.
# Recognition confidence naturally dips for a frame or two while walking
# (motion blur, head turn, partial profile) — without this, a box's name
# flickered to Unknown on every such dip even though it was still clearly
# the same tracked person. 1 = old flicker-prone behavior.
UNKNOWN_STREAK_TO_DROP = 15  # many more consecutive unknowns needed before name drops
MAX_DISPLAY_WIDTH     = 1280
MAX_DISPLAY_HEIGHT    = 720
RENDER_TARGET_FPS     = 120
_RENDER_INTERVAL      = 1.0 / RENDER_TARGET_FPS
_CAMERA_IDLE_SLEEP    = 0.001

# ── Re-ID (occlusion / walk-away recovery) ───────────────────────────────────
# When a face is lost (no spatial match for IDENTITY_PERSISTENCE_TIMEOUT),
# its embedding is kept here briefly so that if the SAME person reappears —
# after turning away, being occluded, or walking off-frame and back — they
# resume their existing track_id (no re-greet, recognition history stays
# continuous) instead of being treated as a brand-new face.
REID_TTL_SECONDS          = 15.0  # keep lost embedding longer — matches new FADE_OUT_TIME
REID_SIMILARITY_THRESHOLD = 0.55  # cosine sim required to resume a lost track (raised from 0.50 — re-ID errors are silent and compound, so this needs to be stricter than live recognition, not looser)
REID_MARGIN               = 0.07  # best lost-track match must beat 2nd-best by this much, same rationale as IDENTITY_MARGIN

