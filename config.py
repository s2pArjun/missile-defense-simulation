# =============================================================================
# config.py — Central Configuration & Settings for Missile Defense Simulation
# =============================================================================

class Config:
    """
    Central configuration class. All toggles, physics constants, and simulation
    parameters live here. Modify this file to tune the simulation behaviour.
    """

    # ── Time ─────────────────────────────────────────────────────────────────
    DT            = 0.03      # Simulation time-step (seconds)
    MAX_TIME      = 30.0      # Maximum simulation duration (seconds)

    # ── World ────────────────────────────────────────────────────────────────
    WORLD_WIDTH   = 1050.0    # Horizontal extent of the battlefield (m)
    WORLD_HEIGHT  =  620.0    # Vertical extent (m)
    GROUND_Y      =    0.0    # Ground level (y = 0)

    # ── Physics ──────────────────────────────────────────────────────────────
    GRAVITY       = -9.81     # Gravitational acceleration (m/s²)
    # Small random perturbation added each step to missile acceleration
    ACCEL_NOISE   =  0.1      # ±magnitude (m/s²) — keeps trajectories realistic

    # ── Missiles ─────────────────────────────────────────────────────────────
    ENABLE_MULTI_MISSILE     = True   # Launch more than one missile?
    NUM_MISSILES             = 3      # How many enemy missiles to launch

    # Launch origins (x, y) — all start from the left edge near the ground
    MISSILE_LAUNCH_POSITIONS = [
        (0.0,  10.0),
        (0.0,  10.0),
        (0.0,  10.0),
    ]

    # Initial velocity vectors (vx, vy) in m/s
    # Physics: range = vx * (2*vy / g), so vy = g*range/(2*vx)
    # Missile 1: vx=48, vy=90  → range≈882 m, peak≈413 m
    # Missile 2: vx=42, vy=105 → range≈900 m, peak≈561 m
    # Missile 3: vx=55, vy=80  → range≈898 m, peak≈326 m
    MISSILE_INITIAL_VELOCITIES = [
        (48.0,  90.0),
        (42.0, 105.0),
        (55.0,  80.0),
    ]

    # ── Interceptors ─────────────────────────────────────────────────────────
    INTERCEPTOR_SPEED        = 120.0  # Fixed interceptor speed (m/s)
    INTERCEPTOR_LAUNCH_X     = 600.0  # x-position of the launcher battery
    INTERCEPTOR_LAUNCH_Y     =   5.0  # y-position of the launcher battery

    # ── Interception Logic ───────────────────────────────────────────────────
    INTERCEPT_DISTANCE_THRESHOLD = 22.0  # Proximity (m) to count as a hit
    HIT_PROBABILITY              = 0.85  # Probabilistic hit chance [0, 1]
    INTERCEPT_LOOKAHEAD_STEPS    = 4     # Extra steps to refine intercept point

    # ── Target ───────────────────────────────────────────────────────────────
    ENABLE_MOVING_TARGET     = True   # Static vs moving target
    TARGET_INITIAL_POS       = (870.0, 5.0)   # (x, y)
    TARGET_SPEED             = 12.0            # m/s (if moving)
    TARGET_MOVE_RANGE        = (770.0, 960.0) # x bounds for back-and-forth

    # ── Visualisation ────────────────────────────────────────────────────────
    ANIMATION_INTERVAL       = 30     # ms between animation frames
    FIGURE_SIZE              = (14, 8)
    TRAIL_LENGTH             = 120    # Max trail points kept per entity

    # ── Logging ──────────────────────────────────────────────────────────────
    VERBOSE_LOGGING          = True   # Print detailed event messages