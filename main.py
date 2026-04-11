# =============================================================================
# main.py — Entry Point & Matplotlib Animation
# =============================================================================

import sys
import os

# Ensure the project root is on the path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend — saves PNG/GIF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from config     import Config
from simulation import Simulation


# ── Colour Palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg"            : "#0a0a1a",
    "grid"          : "#1a1a3a",
    "ground"        : "#2d4a1e",
    "ground_line"   : "#4a7c2f",
    "missile_trail" : "#ff4444",
    "missile_dot"   : "#ff8800",
    "intrcpt_trail" : "#00cc44",
    "intrcpt_dot"   : "#88ff44",
    "target_alive"  : "#00ccff",
    "target_dead"   : "#ff0044",
    "intercept_flash": "#ffffff",
    "text"          : "#e0e0ff",
    "title"         : "#ffcc00",
    "hud_bg"        : "#0a0a2a",
}


def build_figure(sim: Simulation):
    """
    Construct and return all Matplotlib artists for the animation.
    Returns (fig, ax, artists_dict).
    """
    fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE,
                           facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    # ── Axes limits & labels ──────────────────────────────────────────────
    ax.set_xlim(-50, Config.WORLD_WIDTH + 50)
    ax.set_ylim(-30, Config.WORLD_HEIGHT + 50)
    ax.set_aspect("equal")
    ax.tick_params(colors=PALETTE["text"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    ax.set_xlabel("Horizontal Distance (m)", color=PALETTE["text"], fontsize=10)
    ax.set_ylabel("Altitude (m)", color=PALETTE["text"], fontsize=10)
    ax.set_title("🛡  Missile Defense Simulation  🛡",
                 color=PALETTE["title"], fontsize=14, fontweight="bold", pad=12)

    # ── Grid ──────────────────────────────────────────────────────────────
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.4, alpha=0.6)

    # ── Ground strip ──────────────────────────────────────────────────────
    ax.axhspan(-30, 0, color=PALETTE["ground"], alpha=0.8, zorder=0)
    ax.axhline(0, color=PALETTE["ground_line"], linewidth=1.5, zorder=1)

    # ── Launcher battery marker ───────────────────────────────────────────
    ax.plot(Config.INTERCEPTOR_LAUNCH_X, Config.INTERCEPTOR_LAUNCH_Y,
            marker="^", color=PALETTE["intrcpt_dot"], markersize=12,
            zorder=5, label="Launcher battery")
    ax.annotate("🛡 Battery",
                xy=(Config.INTERCEPTOR_LAUNCH_X, Config.INTERCEPTOR_LAUNCH_Y),
                xytext=(Config.INTERCEPTOR_LAUNCH_X, 35),
                color=PALETTE["intrcpt_dot"], fontsize=8,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=PALETTE["intrcpt_dot"],
                                lw=0.8))

    # ── Per-missile artists ───────────────────────────────────────────────
    missile_trails, missile_dots, missile_labels = [], [], []
    for i, m in enumerate(sim.missiles):
        trail,  = ax.plot([], [], color=PALETTE["missile_trail"],
                          linewidth=1.2, alpha=0.7, zorder=3)
        dot,    = ax.plot([], [], "o", color=PALETTE["missile_dot"],
                          markersize=8, zorder=4)
        label   = ax.text(0, 0, f"M{m.id}", color=PALETTE["missile_dot"],
                          fontsize=7, zorder=6, ha="center")
        missile_trails.append(trail)
        missile_dots.append(dot)
        missile_labels.append(label)

    # ── Per-interceptor artists ───────────────────────────────────────────
    intrcpt_trails, intrcpt_dots = [], []
    for _ in sim.interceptors:
        trail, = ax.plot([], [], color=PALETTE["intrcpt_trail"],
                         linewidth=1.0, alpha=0.7, linestyle="--", zorder=3)
        dot,   = ax.plot([], [], "D", color=PALETTE["intrcpt_dot"],
                         markersize=6, zorder=4)
        intrcpt_trails.append(trail)
        intrcpt_dots.append(dot)

    # ── Target artist ─────────────────────────────────────────────────────
    target_marker, = ax.plot(*sim.target.position, "s",
                              color=PALETTE["target_alive"],
                              markersize=14, zorder=5, label="Defended asset")
    target_label   = ax.text(*sim.target.position, "  🏙 Target",
                              color=PALETTE["target_alive"], fontsize=9, zorder=6)
    target_ring    = plt.Circle(sim.target.position, 40,
                                color=PALETTE["target_alive"],
                                fill=False, linewidth=0.8, linestyle=":",
                                alpha=0.4, zorder=2)
    ax.add_patch(target_ring)

    # ── HUD text ──────────────────────────────────────────────────────────
    hud = ax.text(0.01, 0.98, "", transform=ax.transAxes,
                  color=PALETTE["text"], fontsize=9, va="top",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.4",
                            facecolor=PALETTE["hud_bg"], alpha=0.75,
                            edgecolor=PALETTE["grid"]))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color=PALETTE["missile_trail"],  linewidth=2, label="Missile trail"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["missile_dot"],    markersize=8, label="Missile"),
        Line2D([0], [0], color=PALETTE["intrcpt_trail"],  linewidth=2,
               linestyle="--", label="Interceptor trail"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=PALETTE["intrcpt_dot"],    markersize=7, label="Interceptor"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=PALETTE["target_alive"],   markersize=9, label="Target"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              facecolor=PALETTE["hud_bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8)

    artists = {
        "missile_trails"  : missile_trails,
        "missile_dots"    : missile_dots,
        "missile_labels"  : missile_labels,
        "intrcpt_trails"  : intrcpt_trails,
        "intrcpt_dots"    : intrcpt_dots,
        "target_marker"   : target_marker,
        "target_label"    : target_label,
        "target_ring"     : target_ring,
        "hud"             : hud,
    }
    return fig, ax, artists


def animate(sim: Simulation, output_path: str = "missile_defense.gif",
            max_frames: int = 400) -> None:
    """
    Run the simulation step-by-step and save the result as an animated GIF.
    """

    # Pre-run simulation and collect all snapshots
    print("[Renderer] Pre-computing simulation frames…")
    snapshots = []
    while sim.running and len(snapshots) < max_frames:
        sim.step()
        snapshots.append(sim.get_render_snapshot())

    # Also grab final state
    snapshots.append(sim.get_render_snapshot())

    print(f"[Renderer] Captured {len(snapshots)} frames. Building figure…")
    fig, ax, art = build_figure(sim)

    # We'll manage interceptor artists dynamically since they spawn mid-sim
    intrcpt_trail_cache: dict = {}
    intrcpt_dot_cache:   dict = {}

    def _get_or_create_intrcpt_artists(idx):
        if idx not in intrcpt_trail_cache:
            trail, = ax.plot([], [], color=PALETTE["intrcpt_trail"],
                             linewidth=1.0, alpha=0.8, linestyle="--", zorder=3)
            dot,   = ax.plot([], [], "D", color=PALETTE["intrcpt_dot"],
                             markersize=6, zorder=4)
            intrcpt_trail_cache[idx] = trail
            intrcpt_dot_cache[idx]   = dot
        return intrcpt_trail_cache[idx], intrcpt_dot_cache[idx]

    def update_frame(frame_idx):
        snap = snapshots[frame_idx]
        t    = snap["time"]

        # ── Missiles ──────────────────────────────────────────────────────
        for i, (pos, tx, ty, active, intercepted) in enumerate(snap["missiles"]):
            trail = art["missile_trails"][i]
            dot   = art["missile_dots"][i]
            lbl   = art["missile_labels"][i]

            if tx:
                trail.set_data(tx, ty)
            if active:
                dot.set_data([pos[0]], [pos[1]])
                dot.set_visible(True)
                lbl.set_position((pos[0], pos[1] + 15))
                lbl.set_visible(True)
                color = PALETTE["missile_dot"]
            else:
                dot.set_visible(False)
                lbl.set_visible(False)
                # Flash trail briefly on intercept
                trail.set_alpha(0.3 if intercepted else 0.15)

        # ── Interceptors (dynamic — may spawn mid-sim) ────────────────────
        for i, (pos, tx, ty, active, hit) in enumerate(snap["interceptors"]):
            trail_art, dot_art = _get_or_create_intrcpt_artists(i)

            if tx:
                trail_art.set_data(tx, ty)
            if active:
                dot_art.set_data([pos[0]], [pos[1]])
                dot_art.set_visible(True)
                dot_art.set_color(PALETTE["intrcpt_dot"])
            else:
                dot_art.set_visible(False)
                trail_art.set_alpha(0.2)

        # ── Target ────────────────────────────────────────────────────────
        tpos, talive = snap["target"]
        art["target_marker"].set_data([tpos[0]], [tpos[1]])
        art["target_ring"].center = (tpos[0], tpos[1])
        art["target_label"].set_position((tpos[0] + 5, tpos[1] + 20))

        if not talive:
            art["target_marker"].set_color(PALETTE["target_dead"])
            art["target_label"].set_text("  💥 DESTROYED")
            art["target_label"].set_color(PALETTE["target_dead"])
            art["target_ring"].set_edgecolor(PALETTE["target_dead"])

        # ── HUD ───────────────────────────────────────────────────────────
        intercepted_count = sum(1 for _, _, _, _, icp in snap["missiles"] if icp)
        active_count      = sum(1 for _, _, _, act, _ in snap["missiles"] if act)
        active_intrcpt    = sum(1 for _, _, _, act, _ in snap["interceptors"] if act)

        status_str = "🟢 RUNNING" if snap["running"] else "🔴 ENDED"
        hud_text = (
            f"T = {t:6.2f} s          {status_str}\n"
            f"Missiles active   : {active_count}\n"
            f"Intercepted       : {intercepted_count}\n"
            f"Interceptors live : {active_intrcpt}\n"
            f"Target            : {'✅ SAFE' if talive else '💥 HIT'}"
        )
        art["hud"].set_text(hud_text)

        return []   # FuncAnimation requires iterable return

    print("[Renderer] Encoding animation (this may take a moment)…")
    ani = FuncAnimation(fig, update_frame,
                        frames=len(snapshots),
                        interval=Config.ANIMATION_INTERVAL,
                        blit=False)

    ani.save(output_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"[Renderer] Animation saved → {output_path}")


# ── Also save a static summary figure ────────────────────────────────────────

def save_static_summary(sim: Simulation, output_path: str = "missile_defense_summary.png") -> None:
    """
    Save a static PNG showing full trajectory trails and event labels.
    """
    fig, ax = plt.subplots(figsize=Config.FIGURE_SIZE, facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(-50, Config.WORLD_WIDTH + 50)
    ax.set_ylim(-30, Config.WORLD_HEIGHT + 50)
    ax.set_aspect("equal")
    ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.4, alpha=0.5)
    ax.tick_params(colors=PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["grid"])

    ax.axhspan(-30, 0, color=PALETTE["ground"], alpha=0.8)
    ax.axhline(0, color=PALETTE["ground_line"], linewidth=1.5)

    ax.set_xlabel("Horizontal Distance (m)", color=PALETTE["text"], fontsize=10)
    ax.set_ylabel("Altitude (m)", color=PALETTE["text"], fontsize=10)
    ax.set_title("Missile Defense — Full Trajectory Summary",
                 color=PALETTE["title"], fontsize=13, fontweight="bold")

    colors_m = ["#ff4444", "#ff8800", "#ff44aa"]
    colors_i = ["#00cc44", "#44ffaa", "#00aaff"]

    for idx, m in enumerate(sim.missiles):
        c = colors_m[idx % len(colors_m)]
        if m.history:
            xs = [p[0] for p in m.history]
            ys = [p[1] for p in m.history]
            ax.plot(xs, ys, color=c, linewidth=1.5, alpha=0.8,
                    label=f"Missile #{m.id}")
            ax.plot(xs[0], ys[0], "o", color=c, markersize=7)
            status = "💥 Intercepted" if m.intercepted else "🔴 Impacted"
            ax.plot(xs[-1], ys[-1], "x" if m.intercepted else "v",
                    color=c, markersize=10, markeredgewidth=2)
            ax.annotate(f"M{m.id} {status}",
                        xy=(xs[-1], ys[-1]),
                        xytext=(xs[-1] + 20, ys[-1] + 30),
                        color=c, fontsize=7,
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.7))

    for idx, intr in enumerate(sim.interceptors):
        c = colors_i[idx % len(colors_i)]
        if intr.history:
            xs = [p[0] for p in intr.history]
            ys = [p[1] for p in intr.history]
            ax.plot(xs, ys, color=c, linewidth=1.2, linestyle="--",
                    alpha=0.75, label=f"Interceptor #{intr.id}")
            ax.plot(xs[-1], ys[-1], "D", color=c, markersize=7)

    # Target
    t = sim.target
    ax.plot(t.position[0], t.position[1], "s",
            color=PALETTE["target_alive"] if t.alive else PALETTE["target_dead"],
            markersize=14, zorder=5)
    ax.annotate("Target", xy=(t.position[0], t.position[1]),
                xytext=(t.position[0], t.position[1] + 45),
                color=PALETTE["target_alive"] if t.alive else PALETTE["target_dead"],
                fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.8,
                                color=(PALETTE["target_alive"] if t.alive
                                       else PALETTE["target_dead"])))

    ax.legend(loc="upper right", facecolor=PALETTE["hud_bg"],
              edgecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"[Renderer] Summary figure saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)   # Reproducible run

    # 1. Instantiate and run simulation (collecting snapshots inside animate())
    sim = Simulation()

    # 2. Produce animated GIF
    animate(sim, output_path="missile_defense.gif",
            max_frames=350)

    # 3. Produce static summary PNG
    save_static_summary(sim,
                        output_path="missile_defense_summary.png")

    # 4. Print final event log
    print("\n── Event Log ──────────────────────────────────────────────")
    for ev in sim.events:
        print(f"  [{ev['sim_time']:6.2f}s] {ev['type']}", end="")
        if "missile_id"      in ev: print(f"  missile={ev['missile_id']}", end="")
        if "interceptor_id"  in ev: print(f"  interceptor={ev['interceptor_id']}", end="")
        if "impact_pos"      in ev:
            p = ev['impact_pos']
            print(f"  impact=({p[0]:.1f},{p[1]:.1f})", end="")
        if "position"        in ev:
            p = ev['position']
            print(f"  pos=({p[0]:.1f},{p[1]:.1f})", end="")
        print()
    print("───────────────────────────────────────────────────────────\n")