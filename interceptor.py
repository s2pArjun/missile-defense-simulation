# =============================================================================
# interceptor.py — Defensive Interceptor Entity
# =============================================================================

import numpy as np
from config import Config
from utils import (vec2, normalize, distance, magnitude,
                   compute_intercept_time, predict_position_at, log)


class Interceptor:
    """
    A smart defensive interceptor that uses predictive targeting to destroy
    incoming enemy missiles.

    Targeting Logic
    ───────────────
    1. At launch, solve the interception geometry to find the earliest point
       in space-time where the interceptor can meet the missile.
    2. Fly at constant speed toward that predicted intercept point.
    3. Each step, optionally re-evaluate whether to correct course (adaptive).
    4. Declare success when distance to the missile ≤ INTERCEPT_DISTANCE_THRESHOLD
       AND a probabilistic hit roll passes.

    State
    ─────
    position        : np.ndarray – current position (m)
    velocity        : np.ndarray – current velocity vector (m/s)
    target_missile  : Missile    – the missile being hunted
    intercept_point : np.ndarray – predicted meeting point
    active          : bool
    hit             : bool       – True if interception succeeded
    """

    _id_counter = 0

    def __init__(self,
                 launch_pos: tuple,
                 target_missile,          # Missile instance
                 interceptor_id: int = None):
        """
        Parameters
        ----------
        launch_pos      : (x, y) of the launch battery
        target_missile  : the Missile object to intercept
        interceptor_id  : optional label
        """
        Interceptor._id_counter += 1
        self.id = interceptor_id if interceptor_id is not None else Interceptor._id_counter

        self.launch_pos    = vec2(*launch_pos)
        self.position      = vec2(*launch_pos)
        self.target_missile = target_missile

        self.speed         = Config.INTERCEPTOR_SPEED
        self.active        = True
        self.hit           = False
        self.miss          = False
        self.time_elapsed  = 0.0

        # Trajectory history for rendering
        self.history: list[np.ndarray] = [self.position.copy()]

        # ── Compute initial intercept point ───────────────────────────────
        self.intercept_point = None
        self.intercept_time  = None

        if self.intercept_point is not None:
            log(f"Interceptor #{self.id} launched → targeting Missile "
                f"#{target_missile.id} | predicted intercept at "
                f"({self.intercept_point[0]:.1f}, {self.intercept_point[1]:.1f}) "
                f"in {self.intercept_time:.2f} s")
        else:
            log(f"Interceptor #{self.id}: No valid intercept solution — "
                f"flying toward missile current position.")
            # Fallback: aim at current missile position
            self.intercept_time  = None

        # ── Compute initial velocity vector ───────────────────────────────
        # Start by pointing toward missile
        direction = normalize(self.target_missile.position - self.position)
        self.velocity = direction * self.speed

    # ── Geometry Solver ───────────────────────────────────────────────────────

    def _solve_intercept(self) -> tuple:
        """
        Delegate to the analytical intercept solver in utils.
        Returns (intercept_position, intercept_time).
        """
        m = self.target_missile
        return compute_intercept_time(
            launcher_pos      = self.launch_pos,
            interceptor_speed = self.speed,
            missile_pos0      = m.position.copy(),
            missile_vel       = m.velocity.copy(),
            gravity           = Config.GRAVITY,
        )

    def _velocity_toward(self, target_pos: np.ndarray) -> np.ndarray:
        """Compute the constant-speed velocity vector aimed at target_pos."""
        direction = normalize(target_pos - self.position)
        return direction * self.speed

    # ── Physics Update ────────────────────────────────────────────────────────

    def update(self, dt: float = Config.DT) -> None:
        if not self.active:
            return

        m = self.target_missile

        # ── If missile already gone → stop ───────────────────────────────
        if not m.active:
            self.active = False
            return

        # ── Proportional Navigation Guidance ─────────────────────────────
        rel_pos = m.position - self.position
        dist = magnitude(rel_pos)

        # Safety: avoid division issues
        if dist < 1e-6:
            return

        rel_vel = m.velocity - self.velocity

        los_dir = rel_pos / dist

        closing_speed = -np.dot(rel_vel, los_dir)

        N = 3.0  # navigation constant (tune 2.5–4)

        # lateral component (perpendicular to LOS)
        lateral = rel_vel - np.dot(rel_vel, los_dir) * los_dir

        acc_cmd = N * closing_speed * lateral

        # ── Acceleration limit (VERY IMPORTANT) ──────────────────────────
        max_acc = 300.0
        acc_mag = magnitude(acc_cmd)
        if acc_mag > max_acc:
            acc_cmd = (acc_cmd / acc_mag) * max_acc

        # ── Update velocity ──────────────────────────────────────────────
        self.velocity += acc_cmd * dt

        # Keep constant speed
        vel_mag = magnitude(self.velocity)
        if vel_mag > 1e-6:
            self.velocity = (self.velocity / vel_mag) * self.speed

        # ── Move ─────────────────────────────────────────────────────────
        self.position = self.position + self.velocity * dt

        # ── Record trail ─────────────────────────────────────────────────
        self.history.append(self.position.copy())
        if len(self.history) > Config.TRAIL_LENGTH:
            self.history.pop(0)

        self.time_elapsed += dt

        # ── Hit check ────────────────────────────────────────────────────
        dist_to_missile = distance(self.position, m.position)

        if dist_to_missile <= Config.INTERCEPT_DISTANCE_THRESHOLD:
            if np.random.random() <= Config.HIT_PROBABILITY:
                self.hit = True
                self.active = False
                m.mark_intercepted()
                log(f"✅ INTERCEPT SUCCESS — Interceptor #{self.id} "
                    f"destroyed Missile #{m.id} at "
                    f"({self.position[0]:.1f}, {self.position[1]:.1f})")
            else:
                self.miss = True
                self.active = False
                log(f"❌ INTERCEPT MISS (probability fail) — "
                    f"Interceptor #{self.id} vs Missile #{m.id}")

        # ── Boundary check ───────────────────────────────────────────────
        if (self.position[1] < Config.GROUND_Y or
            self.position[0] < 0 or
            self.position[0] > Config.WORLD_WIDTH * 1.2 or
            self.position[1] > Config.WORLD_HEIGHT * 1.5):
            self.active = False
        # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def trail_x(self) -> list:
        return [p[0] for p in self.history]

    @property
    def trail_y(self) -> list:
        return [p[1] for p in self.history]

    def __repr__(self) -> str:
        status = "hit" if self.hit else ("miss" if self.miss else
                 ("active" if self.active else "inactive"))
        return (f"Interceptor(id={self.id}, status={status}, "
                f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}))")