# =============================================================================
# interceptor.py — Defensive Interceptor Entity (Physics-Based)
# =============================================================================
#
# PHYSICS MODEL
# ─────────────
# The interceptor is a rocket that obeys Newton's Laws at every time-step,
# just like the enemy missile.  Its total acceleration each step is:
#
#   a_total = a_gravity                 <- always acts (9.81 m/s^2 downward)
#           + a_thrust_bias             <- sustainer motor cancels gravity
#                                          during boost phase; zero after burnout
#           + a_PN                      <- Augmented Proportional Navigation
#                                          lateral steering command
#
# After burnout the motor is off, so only gravity + PN corrections act on the
# body — the interceptor follows a true ballistic arc, just like a real
# hit-to-kill kinetic warhead (THAAD, Arrow, Patriot PAC-3).
#
# Position & velocity use the same kinematic equations as the Missile class:
#   v(t+dt) = v(t) + a_total * dt
#   x(t+dt) = x(t) + v(t)*dt + 0.5*a_total*dt^2
#
# =============================================================================

import numpy as np
from config import Config
from utils  import (vec2, normalize, distance, magnitude,
                    compute_intercept_time, log)


class Interceptor:
    """
    Physics-based smart interceptor missile.

    Motion:   Full Newtonian integration under gravity + rocket thrust.
    Guidance: Augmented Proportional Navigation (APN) — the same guidance law
              used in Patriot PAC-3, THAAD, and Arrow interceptors.

    Lifecycle phases
    ----------------
    1. Boost  (0 -> burnout_time):
          Sustainer motor fires; thrust offsets gravity so the rocket climbs.
          APN steering curves it toward the predicted intercept point.

    2. Coast  (burnout_time -> intercept):
          Motor off.  Interceptor falls under gravity like a ballistic body.
          Small residual APN corrections via attitude thrusters keep it on track.

    Attributes
    ----------
    position        np.ndarray [x, y]  current position  (m)
    velocity        np.ndarray [vx,vy] current velocity  (m/s)
    acceleration    np.ndarray         last total accel  (m/s^2)
    intercept_point np.ndarray         predicted meet point
    intercept_time  float              seconds to predicted intercept
    active          bool
    hit             bool
    miss            bool
    """

    _id_counter = 0

    def __init__(self,
                 launch_pos: tuple,
                 target_missile,
                 interceptor_id: int = None):
        Interceptor._id_counter += 1
        self.id = (interceptor_id
                   if interceptor_id is not None
                   else Interceptor._id_counter)

        self.launch_pos     = vec2(*launch_pos)
        self.position       = vec2(*launch_pos)
        self.target_missile = target_missile

        self.speed         = Config.INTERCEPTOR_SPEED
        self.active        = True
        self.hit           = False
        self.miss          = False
        self.time_elapsed  = 0.0
        self.acceleration  = vec2(0.0, 0.0)

        # Trajectory history for rendering
        self.history: list = [self.position.copy()]

        # --- Solve intercept geometry ----------------------------------------
        self.intercept_point, self.intercept_time = self._solve_intercept()

        if self.intercept_point is not None:
            log(f"Interceptor #{self.id} launched -> Missile "
                f"#{target_missile.id} | intercept predicted at "
                f"({self.intercept_point[0]:.1f}, {self.intercept_point[1]:.1f}) "
                f"in {self.intercept_time:.2f} s")
        else:
            # Fallback: aim at missile's current position
            self.intercept_point = target_missile.position.copy()
            self.intercept_time  = 8.0
            log(f"Interceptor #{self.id}: no analytic solution "
                f"— fallback to missile current position.")

        # --- Initial velocity: fire toward predicted intercept point ----------
        # WHY 1.4x: with real gravity now acting on the body, a rocket fired
        # at exactly INTERCEPTOR_SPEED will fall short — gravity bleeds off
        # vertical speed continuously.  The 1.4x factor pre-compensates for
        # those gravity losses over the boost phase so the rocket still reaches
        # the predicted intercept point despite being pulled downward.
        direction        = normalize(self.intercept_point - self.position)
        direction[1]    += 2.0          # push y component upward strongly
        direction        = normalize(direction)   # renormalize to unit vector

        launch_speed  = self.speed * 2.2          # compensate for gravity losses

        self.velocity = direction * launch_speed

        # WHY 60% cap: 75% was too long for fast early intercepts — the motor
        # was still firing after the rocket had already passed the intercept
        # window, wasting thrust and extending the trajectory past the target.
        # 60% gives enough boost to reach altitude, then coasts ballistically.
        self._burnout_time = (min(self.intercept_time * 0.7, 6.0)
                              if self.intercept_time else 5.0)

    # ── Geometry Solver ───────────────────────────────────────────────────────

    def _solve_intercept(self) -> tuple:
        """Use utils analytical solver to find (intercept_pos, intercept_time)."""
        m = self.target_missile
        return compute_intercept_time(
            launcher_pos      = self.launch_pos,
            interceptor_speed = self.speed,
            missile_pos0      = m.position.copy(),
            missile_vel       = m.velocity.copy(),
            gravity           = Config.GRAVITY,
        )

    # ── Physics Update ────────────────────────────────────────────────────────

    def update(self, dt: float = Config.DT) -> None:
        """
        Advance interceptor state by one time-step using full Newtonian physics.

        Acceleration budget
        -------------------
        a_gravity     = (0, g)           <- Newton's law; always present
        a_thrust_bias = (0, -g * k)      <- sustainer thrust (boost phase only)
                      = (0,  0)          <- zero after burnout (coast phase)
        a_PN          = APN command      <- Augmented PN lateral steering

        a_total = a_gravity + a_thrust_bias + a_PN

        Kinematic integration (identical to Missile.update):
            v(t+dt) = v(t) + a_total * dt
            x(t+dt) = x(t) + v(t)*dt + 0.5*a_total*dt^2
        """
        if not self.active:
            return

        m = self.target_missile

        # Target already gone -> self-destruct
        if not m.active:
            self.active = False
            return

        # =================================================================
        # 1. GRAVITY — Newton's 2nd Law; acts on every mass every step
        # =================================================================
        # Config.GRAVITY = -9.81  (negative = downward)
        a_gravity = vec2(0.0, Config.GRAVITY)

        # =================================================================
        # 2. SUSTAINER THRUST (boost phase only)
        # =================================================================
        # WHY abs(GRAVITY): Config.GRAVITY = -9.81.  The old code did
        # -Config.GRAVITY * thrust_mult = +9.81 * 1.55 = +15.2 m/s² upward.
        # That seems right, BUT at burnout thrust_mult = 1.0, so thrust =
        # +9.81 which exactly cancels gravity -> net accel = 0, not falling.
        # Then motor cuts to zero and suddenly full gravity hits -> kink.
        #
        # The fix: be explicit.  Thrust = abs(g) to cancel gravity PLUS an
        # extra "net_climb" term that tapers from +30 m/s² at ignition to 0
        # at burnout.  At burnout thrust = abs(g) so net = 0, then cuts to 0
        # smoothly — no discontinuous jump.  After burnout: pure ballistic.
        if self.time_elapsed < self._burnout_time:
            burn_frac = 1.0 - (self.time_elapsed / self._burnout_time)
            # Extra upward acceleration on top of gravity cancellation
            net_climb = 30.0 * burn_frac          # 30 m/s² at ignition -> 0 at burnout
            # Total upward thrust = cancel gravity + provide net climb
            a_thrust = vec2(0.0, abs(Config.GRAVITY) + net_climb)
        else:
            # Motor exhausted: interceptor coasts under pure gravity
            a_thrust = vec2(0.0, 0.0)

        # =================================================================
        # 3. AUGMENTED PROPORTIONAL NAVIGATION (APN)
        # =================================================================
        # Classic PN:  a_cmd = N * Vc * omega_perp
        # Augmented PN adds a feedforward term for target acceleration:
        #              a_cmd += (N/2) * a_target
        #
        # Variables:
        #   R     = relative position vector (missile pos - interceptor pos)
        #   Vc    = closing speed (rate range decreases; positive = closing)
        #   omega = LOS angular velocity (rad/s) — 2-D scalar
        #   N     = navigation ratio (3-5 typical; higher = tighter tracking)
        #   a_target = acceleration of the tracked missile = gravity vector
        # =================================================================
        a_PN = vec2(0.0, 0.0)

        rel_pos = m.position - self.position    # R vector
        dist    = magnitude(rel_pos)

        if dist > 0.5:                          # avoid singularity at contact
            los_unit = rel_pos / dist           # unit LOS vector

            # Relative velocity of missile w.r.t. interceptor
            rel_vel = m.velocity - self.velocity

            # Closing speed (positive when target is approaching)
            Vc = -np.dot(rel_vel, los_unit)

            # LOS angular rate (2-D cross product divided by range squared)
            # omega = (R x V_rel) / |R|^2
            omega = ((rel_pos[0] * rel_vel[1] - rel_pos[1] * rel_vel[0])
                     / (dist ** 2))

            # Unit vector perpendicular to LOS (90 deg CCW of los_unit)
            los_perp = vec2(-los_unit[1], los_unit[0])

            N = 4.5   # navigation ratio

            # Pure PN steering term — always active
            # Steers the interceptor to zero the LOS rotation rate
            a_PN = N * Vc * omega * los_perp

            # WHY coast-phase only: during boost the interceptor is climbing
            # hard and the aug_term (0, -22 m/s²) was pulling it DOWN at
            # exactly the moment it needed to be going UP — fighting the thrust.
            # The augmentation only becomes meaningful when the missile is in
            # its falling phase and the interceptor is coasting to meet it.
            #
            # WHY m.acceleration not vec2(0, GRAVITY): m.acceleration is
            # updated each step to include noise, so this tracks the missile's
            # TRUE instantaneous accel rather than assuming pure gravity.
            if self.time_elapsed >= self._burnout_time:
                a_PN += (N / 2.0) * m.acceleration    # feedforward for target accel

            # --- Lateral acceleration limit (manoeuvre envelope) -----------
            # Real interceptors are limited to ~20-40 g lateral manoeuvre.
            # Capping prevents unrealistic instant turns.
            MAX_LAT_G = 22.0 * abs(Config.GRAVITY)  # ~216 m/s^2
            pn_mag = magnitude(a_PN)
            if pn_mag > MAX_LAT_G:
                a_PN = (a_PN / pn_mag) * MAX_LAT_G

        # =================================================================
        # 4. TOTAL ACCELERATION (Newton's Second Law: F = m*a)
        # =================================================================
        a_total = a_gravity + a_thrust + a_PN
        self.acceleration = a_total.copy()

        # =================================================================
        # 5. KINEMATIC INTEGRATION (same equations as Missile.update)
        # =================================================================
        # Position:  x(t+dt) = x(t) + v(t)*dt + 0.5*a*dt^2
        self.position = (self.position
                         + self.velocity * dt
                         + 0.5 * a_total * (dt ** 2))

        # Velocity:  v(t+dt) = v(t) + a*dt
        self.velocity = self.velocity + a_total * dt

        # =================================================================
        # 6. AERODYNAMIC DRAG  (F_drag proportional to v^2)
        # =================================================================
        # Simplified model: a_drag = -k * |v| * v
        # This decelerates the rocket realistically during coast phase.
        DRAG_K = 0.0005
        spd = magnitude(self.velocity)
        if spd > 1e-6:
            a_drag = -DRAG_K * spd * self.velocity
            self.velocity = self.velocity + a_drag * dt

        # =================================================================
        # 7. TRAIL RECORDING
        # =================================================================
        self.history.append(self.position.copy())
        if len(self.history) > Config.TRAIL_LENGTH:
            self.history.pop(0)

        self.time_elapsed += dt

        # =================================================================
        # 8. PROXIMITY / HIT CHECK
        # =================================================================
        if m.active:
            dist_to_missile = distance(self.position, m.position)

            if dist_to_missile <= Config.INTERCEPT_DISTANCE_THRESHOLD:
                # Probabilistic hit roll (simulates warhead reliability)
                if np.random.random() <= Config.HIT_PROBABILITY:
                    self.hit    = True
                    self.active = False
                    m.mark_intercepted()
                    log(f"[HIT]  Interceptor #{self.id} killed Missile "
                        f"#{m.id} at ({self.position[0]:.1f}, "
                        f"{self.position[1]:.1f}) m  "
                        f"t={self.time_elapsed:.2f} s")
                else:
                    self.miss   = True
                    self.active = False
                    log(f"[MISS] Interceptor #{self.id} passed Missile "
                        f"#{m.id} — probabilistic failure")

        # =================================================================
        # 9. BOUNDARY / GROUND DEACTIVATION
        # =================================================================
        if (self.position[1] < Config.GROUND_Y or
                self.position[0] < -50 or
                self.position[0] > Config.WORLD_WIDTH * 1.3 or
                self.position[1] > Config.WORLD_HEIGHT * 1.6):
            self.active = False

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def trail_x(self) -> list:
        return [p[0] for p in self.history]

    @property
    def trail_y(self) -> list:
        return [p[1] for p in self.history]

    def __repr__(self) -> str:
        status = ("hit"    if self.hit  else
                  "miss"   if self.miss else
                  "active" if self.active else "inactive")
        return (f"Interceptor(id={self.id}, status={status}, "
                f"pos=({self.position[0]:.1f}, {self.position[1]:.1f}), "
                f"vel=({self.velocity[0]:.1f}, {self.velocity[1]:.1f}))")