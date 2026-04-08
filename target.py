# =============================================================================
# target.py — Defended Asset (Static or Moving)
# =============================================================================

import numpy as np
from config import Config
from utils import vec2, log


class Target:
    """
    Represents the asset being defended (a city, military base, etc.).

    Movement Modes
    ──────────────
    • Static  : Target remains at its initial position.
    • Moving  : Target performs linear back-and-forth patrol within
                TARGET_MOVE_RANGE, with a small random jitter each step
                to simulate imperfect movement (convoy, ship, etc.).

    State
    ─────
    position  : np.ndarray [x, y]
    velocity  : np.ndarray [vx, vy]  (zero if static)
    alive     : bool – False after being struck by a missile
    history   : trail of positions for rendering
    """

    def __init__(self):
        self.position = vec2(*Config.TARGET_INITIAL_POS)
        self.alive    = True
        self.moving   = Config.ENABLE_MOVING_TARGET
        self.speed    = Config.TARGET_SPEED
        self.history: list[np.ndarray] = [self.position.copy()]

        # Direction sign: +1 = moving right, −1 = moving left
        self._direction = 1.0
        self.velocity   = vec2(self.speed * self._direction, 0.0) if self.moving else vec2(0.0, 0.0)

        self._x_min, self._x_max = Config.TARGET_MOVE_RANGE

        mode = "moving" if self.moving else "static"
        log(f"Target spawned ({mode}) at {tuple(self.position)}")

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float = Config.DT) -> None:
        """
        Advance target state by one time-step.

        For a moving target:
        • Translate by velocity × dt.
        • Bounce direction at x-boundary limits.
        • Add tiny random lateral jitter (simulates imperfect motion).
        """
        if not self.alive or not self.moving:
            return

        # Small random jitter on velocity (±5 % of speed)
        jitter = np.random.uniform(-0.05 * self.speed, 0.05 * self.speed)
        self.velocity[0] = self._direction * self.speed + jitter

        # Translate position
        self.position += self.velocity * dt

        # Bounce at patrol boundaries
        if self.position[0] >= self._x_max:
            self.position[0]  = self._x_max
            self._direction   = -1.0
        elif self.position[0] <= self._x_min:
            self.position[0]  = self._x_min
            self._direction   =  1.0

        # Keep on the ground
        self.position[1] = Config.GROUND_Y + Config.TARGET_INITIAL_POS[1]

        # Trail
        self.history.append(self.position.copy())
        if len(self.history) > Config.TRAIL_LENGTH:
            self.history.pop(0)

    # ── Collision ─────────────────────────────────────────────────────────────

    def check_hit(self, missiles: list) -> bool:
        """
        Check whether any active missile has reached the target's vicinity.
        Returns True (and marks target dead) if struck.
        """
        for m in missiles:
            if not m.active and not m.intercepted:
                # Missile impacted ground — check x proximity to target
                if abs(m.position[0] - self.position[0]) < 40.0:
                    self.alive = False
                    log(f"💥 TARGET DESTROYED by Missile #{m.id}!")
                    return True
        return False

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def trail_x(self) -> list:
        return [p[0] for p in self.history]

    @property
    def trail_y(self) -> list:
        return [p[1] for p in self.history]

    def __repr__(self) -> str:
        status = "alive" if self.alive else "DESTROYED"
        return (f"Target(status={status}, "
                f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}))")