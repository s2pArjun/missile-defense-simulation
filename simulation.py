# =============================================================================
# simulation.py — Master Simulation Controller
# =============================================================================

import numpy as np
from config    import Config
from missile   import Missile
from interceptor import Interceptor
from target    import Target
from utils     import predict_impact_point, log


class Simulation:
    """
    Orchestrates the full missile-defense scenario.

    Responsibilities
    ────────────────
    • Spawn all enemy missiles at t=0.
    • Spawn a defensive interceptor for each missile shortly after launch
      (simulating radar lock-on delay).
    • Advance all entities each time-step via their update() methods.
    • Detect target hits and interceptions.
    • Maintain a structured event log for post-simulation analysis.
    • Provide data snapshots each step for the visualiser to consume.

    Attributes
    ──────────
    missiles      : list[Missile]
    interceptors  : list[Interceptor]
    target        : Target
    time          : float  — elapsed simulation time (s)
    running       : bool
    events        : list[dict]  — structured event log
    """

    def __init__(self):
        self.time         = 0.0
        self.step_count   = 0
        self.running      = True
        self.events: list[dict] = []

        # ── Spawn entities ────────────────────────────────────────────────
        self.target = Target()
        self.missiles: list[Missile]      = []
        self.interceptors: list[Interceptor] = []

        self._spawn_missiles()

        log("=" * 60)
        log("Missile Defense Simulation STARTED")
        log(f"Missiles: {len(self.missiles)} | "
            f"Target moving: {Config.ENABLE_MOVING_TARGET}")
        log("=" * 60)

    # ── Spawning ──────────────────────────────────────────────────────────────

    def _spawn_missiles(self) -> None:
        """Create all enemy missiles defined in Config."""
        count = Config.NUM_MISSILES if Config.ENABLE_MULTI_MISSILE else 1
        for i in range(count):
            pos = Config.MISSILE_LAUNCH_POSITIONS[i]
            vel = Config.MISSILE_INITIAL_VELOCITIES[i]
            m   = Missile(launch_pos=pos, launch_vel=vel, missile_id=i + 1)
            self.missiles.append(m)
            self._log_event("missile_launch", missile_id=m.id,
                            position=m.position.copy())

    def _spawn_interceptor_for(self, missile: Missile) -> None:
        """
        Compute the predicted impact point, log it, then launch an interceptor.
        """
        impact_pos, impact_time = predict_impact_point(
            missile.position.copy(), missile.velocity.copy()
        )

        if impact_pos is not None:
            log(f"Radar lock on Missile #{missile.id} — "
                f"predicted impact at ({impact_pos[0]:.1f}, {impact_pos[1]:.1f}) "
                f"in {impact_time:.2f} s")
            self._log_event("predicted_impact",
                            missile_id   = missile.id,
                            impact_pos   = impact_pos.copy(),
                            impact_time  = impact_time)
        else:
            log(f"Could not predict impact for Missile #{missile.id}")

        intercept = Interceptor(
            launch_pos      = (Config.INTERCEPTOR_LAUNCH_X,
                               Config.INTERCEPTOR_LAUNCH_Y),
            target_missile  = missile,
            interceptor_id  = missile.id
        )
        self.interceptors.append(intercept)
        self._log_event("interceptor_launch",
                        interceptor_id  = intercept.id,
                        missile_id      = missile.id,
                        intercept_point = (intercept.intercept_point.copy()
                                           if intercept.intercept_point is not None
                                           else None))

    # ── Main Step ─────────────────────────────────────────────────────────────

    def step(self) -> bool:
        """
        Advance the simulation by one time-step (Config.DT seconds).
        Returns True while the simulation is still running.
        """
        if not self.running:
            return False

        dt = Config.DT

        # ── Update missiles ───────────────────────────────────────────────
        for m in self.missiles:
            m.update(dt)

        # ── Spawn interceptors after a radar-lock delay of ~2.5 s ────────
        # WHY 2.5s: at 0.5 s the missiles have barely left the ground (~45 m
        # altitude).  Intercepting that close to the launcher gives a trivially
        # short trajectory.  At 2.5 s the missiles are mid-arc (~200-350 m
        # altitude) making the intercept geometry far more realistic and the
        # visible animation much more meaningful.
        if self.step_count == int(3.5 / dt):
            for m in self.missiles:
                # Only spawn if not already tracking this missile
                tracked_ids = {intr.target_missile.id
                               for intr in self.interceptors}
                if m.id not in tracked_ids and m.active:
                    self._spawn_interceptor_for(m)

        # ── Update interceptors ───────────────────────────────────────────
        for intr in self.interceptors:
            if intr.active:
                intr.update(dt)
            elif intr.hit:
                self._log_event("intercept_success",
                                interceptor_id = intr.id,
                                missile_id     = intr.target_missile.id,
                                position       = intr.position.copy(),
                                time           = self.time)
                intr.hit = False   # prevent duplicate log entries
            elif intr.miss:
                self._log_event("intercept_miss",
                                interceptor_id = intr.id,
                                missile_id     = intr.target_missile.id,
                                time           = self.time)
                intr.miss = False

        # ── Update target ─────────────────────────────────────────────────
        self.target.update(dt)
        self.target.check_hit(self.missiles)

        # ── Advance time ──────────────────────────────────────────────────
        self.time       += dt
        self.step_count += 1

        # ── Termination checks ────────────────────────────────────────────
        all_missiles_done = all(not m.active for m in self.missiles)
        all_intrcps_done  = all(not i.active for i in self.interceptors)
        time_exceeded     = self.time >= Config.MAX_TIME

        if (all_missiles_done and all_intrcps_done) or time_exceeded:
            self.running = False
            self._print_summary()

        return self.running

    # ── Snapshot for Renderer ─────────────────────────────────────────────────

    def get_render_snapshot(self) -> dict:
        """
        Return a lightweight snapshot of the current state, consumed by the
        Matplotlib animation callback each frame.
        """
        return {
            "time"         : self.time,
            "missiles"     : [(m.position.copy(), m.trail_x[:], m.trail_y[:],
                               m.active, m.intercepted)
                              for m in self.missiles],
            "interceptors" : [(i.position.copy(), i.trail_x[:], i.trail_y[:],
                               i.active, i.hit)
                              for i in self.interceptors],
            "target"       : (self.target.position.copy(), self.target.alive),
            "running"      : self.running,
        }

    # ── Event Log ─────────────────────────────────────────────────────────────

    def _log_event(self, event_type: str, **kwargs) -> None:
        entry = {"type": event_type, "sim_time": self.time, **kwargs}
        self.events.append(entry)

    def _print_summary(self) -> None:
        """Print a formatted end-of-simulation summary."""
        total     = len(self.missiles)
        destroyed = sum(1 for m in self.missiles if m.intercepted)
        impacted  = sum(1 for m in self.missiles if not m.intercepted and not m.active)

        log("=" * 60)
        log("SIMULATION COMPLETE")
        log(f"  Duration         : {self.time:.2f} s")
        log(f"  Missiles launched: {total}")
        log(f"  Intercepted      : {destroyed}")
        log(f"  Reached ground   : {impacted}")
        log(f"  Target status    : {'SURVIVED' if self.target.alive else 'DESTROYED'}")
        log("=" * 60)