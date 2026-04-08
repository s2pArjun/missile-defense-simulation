# =============================================================================
# missile.py — Enemy Missile Entity
# =============================================================================

import numpy as np
from config import Config
from utils import (vec2, position_update, velocity_update,
                   random_accel_noise, log)


class Missile:
    """
    Represents a single ballistic enemy missile.

    Physics model
    ─────────────
    • Gravity is the primary vertical acceleration (constant).
    • A small stochastic noise term is added each time-step to both axes,
      simulating real-world imperfections (wind, thrust variation, etc.).
    • Position and velocity are updated via Newtonian kinematics every dt.

    State
    ─────
    position   : np.ndarray [x, y]   – current position (m)
    velocity   : np.ndarray [vx, vy] – current velocity (m/s)
    acceleration: np.ndarray          – instantaneous acceleration (m/s²)
    history    : list of np.ndarray  – full position trail (for rendering)
    active     : bool                – False once missile hits ground or is intercepted
    """

    # Class-level counter so each missile gets a unique ID
    _id_counter = 0

    def __init__(self,
                 launch_pos: tuple,
                 launch_vel: tuple,
                 missile_id: int = None):
        """
        Parameters
        ----------
        launch_pos : (x, y) launch position in metres
        launch_vel : (vx, vy) initial velocity in m/s
        missile_id : optional integer label
        """
        Missile._id_counter += 1
        self.id = missile_id if missile_id is not None else Missile._id_counter

        # ── Kinematic state ───────────────────────────────────────────────
        self.position     = vec2(*launch_pos)
        self.velocity     = vec2(*launch_vel)
        # Base acceleration = gravity only (noise added dynamically)
        self.acceleration = vec2(0.0, Config.GRAVITY)

        # ── Bookkeeping ───────────────────────────────────────────────────
        self.history: list[np.ndarray] = [self.position.copy()]
        self.active       = True
        self.intercepted  = False
        self.time_elapsed = 0.0

        log(f"Missile #{self.id} launched from {launch_pos} "
            f"with velocity {launch_vel}")

    # ── Physics Update ────────────────────────────────────────────────────────

    def update(self, dt: float = Config.DT) -> None:
        """
        Advance missile state by one time-step dt.

        Steps
        ─────
        1. Compute instantaneous acceleration = gravity + noise
        2. Update position via:  p(t+dt) = p(t) + v·dt + ½·a·dt²
        3. Update velocity via:  v(t+dt) = v(t) + a·dt
        4. Record position in history trail
        5. Check ground collision → deactivate if y ≤ ground level
        """
        if not self.active:
            return

        # 1. Acceleration: gravity (constant) + stochastic perturbation
        noise = random_accel_noise(Config.ACCEL_NOISE)
        self.acceleration = vec2(0.0, Config.GRAVITY) + noise

        # 2. Position update (kinematic equation)
        self.position = position_update(self.position,
                                        self.velocity,
                                        self.acceleration,
                                        dt)

        # 3. Velocity update
        self.velocity = velocity_update(self.velocity,
                                        self.acceleration,
                                        dt)

        # 4. Store trail (cap length for memory efficiency)
        self.history.append(self.position.copy())
        if len(self.history) > Config.TRAIL_LENGTH:
            self.history.pop(0)

        self.time_elapsed += dt

        # 5. Ground check
        if self.position[1] <= Config.GROUND_Y and self.velocity[1] < 0:
            self.position[1] = Config.GROUND_Y
            self.active = False
            log(f"Missile #{self.id} IMPACT at x={self.position[0]:.1f} m "
                f"after {self.time_elapsed:.2f} s")

    # ── State Accessors ───────────────────────────────────────────────────────

    def mark_intercepted(self) -> None:
        """Call this when an interceptor destroys the missile."""
        self.active      = False
        self.intercepted = True

    @property
    def trail_x(self) -> list:
        return [p[0] for p in self.history]

    @property
    def trail_y(self) -> list:
        return [p[1] for p in self.history]

    def __repr__(self) -> str:
        status = ("intercepted" if self.intercepted
                  else "active" if self.active else "impacted")
        return (f"Missile(id={self.id}, status={status}, "
                f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}))")