# =============================================================================
# utils.py — Physics Utilities & Mathematical Helpers
# =============================================================================

import numpy as np
from config import Config


# ── Vector Helpers ────────────────────────────────────────────────────────────

def vec2(x: float, y: float) -> np.ndarray:
    """Create a 2-D NumPy vector."""
    return np.array([x, y], dtype=float)


def magnitude(v: np.ndarray) -> float:
    """Euclidean length of a 2-D vector."""
    return float(np.linalg.norm(v))


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v; returns zero-vector if v is zero."""
    mag = magnitude(v)
    return v / mag if mag > 1e-9 else np.zeros(2)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


# ── Kinematic Equations ───────────────────────────────────────────────────────

def position_update(pos: np.ndarray,
                    vel: np.ndarray,
                    acc: np.ndarray,
                    dt: float) -> np.ndarray:
    """
    Newton's second-law position update:
        x(t+dt) = x(t) + v(t)·dt + ½·a·dt²
    """
    return pos + vel * dt + 0.5 * acc * dt ** 2


def velocity_update(vel: np.ndarray,
                    acc: np.ndarray,
                    dt: float) -> np.ndarray:
    """
    Velocity Verlet update:
        v(t+dt) = v(t) + a·dt
    """
    return vel + acc * dt


# ── Analytical Projectile Solver ──────────────────────────────────────────────

def predict_position_at(pos0: np.ndarray,
                        vel0: np.ndarray,
                        t: float,
                        gravity: float = Config.GRAVITY) -> np.ndarray:
    """
    Analytically predict where a projectile will be at time t seconds from now,
    assuming constant gravity and no further noise.

        x(t) = x0 + vx0·t
        y(t) = y0 + vy0·t + ½·g·t²
    """
    acc = vec2(0.0, gravity)
    return position_update(pos0, vel0, acc, t)


def time_to_ground(pos0: np.ndarray,
                   vel0: np.ndarray,
                   gravity: float = Config.GRAVITY) -> float:
    """
    Solve the quadratic  ½g·t² + vy0·t + y0 = 0  for the first positive root.
    Returns the time (seconds) when the projectile hits y=0, or None if it
    never returns to ground (e.g. already below).

    Quadratic form: a·t² + b·t + c = 0
        a = ½·g   (gravity is negative, so a < 0)
        b = vy0
        c = y0
    """
    a = 0.5 * gravity
    b = float(vel0[1])
    c = float(pos0[1])

    discriminant = b ** 2 - 4.0 * a * c
    if discriminant < 0:
        return None  # Projectile doesn't reach ground

    sqrt_d = np.sqrt(discriminant)
    t1 = (-b + sqrt_d) / (2.0 * a)
    t2 = (-b - sqrt_d) / (2.0 * a)

    # Return the smallest positive root
    candidates = [t for t in (t1, t2) if t > 1e-6]
    return min(candidates) if candidates else None


def predict_impact_point(pos0: np.ndarray,
                         vel0: np.ndarray,
                         gravity: float = Config.GRAVITY) -> tuple:
    """
    Compute (impact_position, impact_time) for a ballistic missile.
    Returns (None, None) if no valid solution exists.
    """
    t_impact = time_to_ground(pos0, vel0, gravity)
    if t_impact is None:
        return None, None

    impact_pos = predict_position_at(pos0, vel0, t_impact, gravity)
    return impact_pos, t_impact


# ── Interception Geometry ─────────────────────────────────────────────────────

def compute_intercept_time(launcher_pos: np.ndarray,
                           interceptor_speed: float,
                           missile_pos0: np.ndarray,
                           missile_vel: np.ndarray,
                           gravity: float = Config.GRAVITY,
                           steps: int = 60) -> tuple:
    """
    Numerically search for the earliest time t* such that the distance between
    the predicted missile position and the launcher position equals
    interceptor_speed × t*.  Implements a discrete search then binary-refines.

    Returns (intercept_position, intercept_time) or (None, None).
    """
    best_t   = None
    best_pos = None
    best_err = float('inf')

    dt_search = min(Config.MAX_TIME, 40.0) / steps

    for i in range(1, steps + 1):
        t = i * dt_search
        missile_future = predict_position_at(missile_pos0, missile_vel, t, gravity)

        if missile_future[1] < Config.GROUND_Y:
            break  # Missile already at/below ground

        # Distance the interceptor must travel to reach that point
        dist_needed   = distance(launcher_pos, missile_future)
        # Distance the interceptor can cover in time t
        dist_possible = interceptor_speed * t

        err = abs(dist_needed - dist_possible)
        if err < best_err:
            best_err   = err
            best_t     = t
            best_pos   = missile_future

    # Binary-refine around best_t for precision
    if best_t is not None:
        lo, hi = max(0.0, best_t - dt_search), best_t + dt_search
        for _ in range(20):
            mid = (lo + hi) / 2.0
            mp  = predict_position_at(missile_pos0, missile_vel, mid, gravity)
            if mp[1] < Config.GROUND_Y:
                hi = mid
                continue
            d_need = distance(launcher_pos, mp)
            d_poss = interceptor_speed * mid
            if d_poss >= d_need:
                hi      = mid
                best_t  = mid
                best_pos = mp
            else:
                lo = mid

    return best_pos, best_t


# ── Noise ─────────────────────────────────────────────────────────────────────

def random_accel_noise(magnitude: float = Config.ACCEL_NOISE) -> np.ndarray:
    """Return a small random 2-D acceleration perturbation."""
    return np.random.uniform(-magnitude, magnitude, size=2)


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Conditional console logger gated by Config.VERBOSE_LOGGING."""
    if Config.VERBOSE_LOGGING:
        print(f"[SIM] {msg}")