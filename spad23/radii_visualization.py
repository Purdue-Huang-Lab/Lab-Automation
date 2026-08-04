import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------- SPAD23 geometry (same as your code) --------------------
pitch_x = 23.0  # µm
pitch_y = 19.92 # µm
rows = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8],
    [9,10,11,12,13],
    [14,15,16,17],
    [18,19,20,21,22],
]
shift_short = True

coords = {}
max_cols = max(len(r) for r in rows)
for r, row in enumerate(rows):
    x0 = (pitch_x/2) if (shift_short and len(row) < max_cols) else 0.0
    for c, pid in enumerate(row):
        coords[pid] = (x0 + c*pitch_x, r*pitch_y)

pids = np.arange(23)
x = np.array([coords[p][0] for p in pids])
y = np.array([coords[p][1] for p in pids])

# Array bounds
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
xc = (xmin + xmax)/2
yc = (ymin + ymax)/2

# -------------------- Gaussian / mapping definitions --------------------
# Your sample-plane half-maximum radii (µm)
r_half_sample = np.array([1.0, 2.0])  # laser-size, diffused

# r_half = sqrt(2 ln 2) * sigma
k = np.sqrt(2*np.log(2))

def I_gauss(r, sig):
    return np.exp(-(r**2)/(2*sig**2))

# -------------------- Interactive state --------------------
state = {
    "x0": xc,
    "y0": yc,
    "M": 20.0,   # default per your request
}

# -------------------- Build figure --------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax_geom = axes[0, 0]
ax_laser = axes[0, 1]
ax_diff  = axes[1, 0]
ax_rad   = axes[1, 1]

# Give room for slider
fig.subplots_adjust(bottom=0.12)

# Geometry plot (with pid labels)
ax_geom.scatter(x, y, s=280, edgecolors='k', c='white', zorder=1)
for pid, xx, yy in zip(pids, x, y):
    ax_geom.text(xx, yy, f"{pid}", ha='center', va='center', fontsize=9, zorder=2)

center_marker_geom = ax_geom.scatter([state["x0"]], [state["y0"]],
                                     marker='x', s=240, linewidths=4, c='tab:blue', zorder=3)
ax_geom.set_title("Click here to set beam center (SPAD plane)")
ax_geom.set_aspect('equal')
ax_geom.set_xlabel("x (µm)")
ax_geom.set_ylabel("y (µm)")
ax_geom.set_xlim(xmin-10, xmax+10)
ax_geom.set_ylim(ymin-10, ymax+10)

# Intensity scatters (initialized; colors updated in update())
sc_laser = ax_laser.scatter(x, y, c=np.zeros_like(x), s=280, edgecolors='k', cmap='inferno')
center_marker_laser = ax_laser.scatter([state["x0"]], [state["y0"]],
                                      marker='x', s=240, linewidths=4, c='k')

sc_diff = ax_diff.scatter(x, y, c=np.zeros_like(x), s=280, edgecolors='k', cmap='inferno')
center_marker_diff = ax_diff.scatter([state["x0"]], [state["y0"]],
                                     marker='x', s=240, linewidths=4, c='k')

ax_laser.set_aspect('equal')
ax_laser.set_xlabel("x (µm)")
ax_laser.set_ylabel("y (µm)")
ax_laser.set_xlim(xmin-10, xmax+10)
ax_laser.set_ylim(ymin-10, ymax+10)

ax_diff.set_aspect('equal')
ax_diff.set_xlabel("x (µm)")
ax_diff.set_ylabel("y (µm)")
ax_diff.set_xlim(xmin-10, xmax+10)
ax_diff.set_ylim(ymin-10, ymax+10)

# Colorbars (fixed scaling 0..1 because we plot normalized Gaussian)
cb1 = fig.colorbar(sc_laser, ax=ax_laser, label="Normalized intensity")
cb2 = fig.colorbar(sc_diff, ax=ax_diff, label="Normalized intensity")
sc_laser.set_clim(0, 1)
sc_diff.set_clim(0, 1)

# Radial plot objects
rad_pts_laser = ax_rad.scatter([], [], s=45, label="laser-size")
rad_pts_diff  = ax_rad.scatter([], [], s=45, label="diffused")
(rad_line_laser,) = ax_rad.plot([], [], linewidth=2)
(rad_line_diff,)  = ax_rad.plot([], [], linewidth=2)

# Reference pitch lines
pitch_line1 = ax_rad.axvline(pitch_x, linestyle='--', linewidth=1)
pitch_line2 = ax_rad.axvline(2*pitch_x, linestyle='--', linewidth=1)

ax_rad.set_xlabel("Radius to beam center on SPAD plane (µm)")
ax_rad.set_ylabel("Normalized intensity")
ax_rad.set_title("23 detectors → 23 radius samples (after off-centering)")
ax_rad.grid(True)
ax_rad.legend()

# Slider for magnification
ax_slider = fig.add_axes([0.18, 0.04, 0.64, 0.03])  # [left, bottom, width, height]
sM = Slider(ax_slider, "Magnification M", 5.0, 50.0, valinit=state["M"], valstep=0.1)

# -------------------- Update function --------------------
def update():
    x0, y0, M = state["x0"], state["y0"], state["M"]

    # radii
    r = np.sqrt((x - x0)**2 + (y - y0)**2)

    # map half-max radii to detector
    r_half_det = M * r_half_sample
    sigma_det = r_half_det / k

    # Intensities
    z_l = I_gauss(r, sigma_det[0])
    z_d = I_gauss(r, sigma_det[1])

    sc_laser.set_array(z_l)
    sc_diff.set_array(z_d)

    # Update center markers
    center_marker_geom.set_offsets([[x0, y0]])
    center_marker_laser.set_offsets([[x0, y0]])
    center_marker_diff.set_offsets([[x0, y0]])

    # Titles with current M and mapped sizes
    ax_laser.set_title(f"M={M:.1f}×  laser-size: r½={r_half_det[0]:.1f} µm  (σ={sigma_det[0]:.1f} µm)")
    ax_diff.set_title(f"M={M:.1f}×  diffused:   r½={r_half_det[1]:.1f} µm  (σ={sigma_det[1]:.1f} µm)")

    # Radial samples (sorted)
    order = np.argsort(r)
    rs = r[order]
    zl = z_l[order]
    zd = z_d[order]

    rad_pts_laser.set_offsets(np.c_[rs, zl])
    rad_pts_diff.set_offsets(np.c_[rs, zd])

    # Smooth curves
    rr = np.linspace(0, max(rs.max()*1.05, 1e-6), 400)
    rad_line_laser.set_data(rr, I_gauss(rr, sigma_det[0]))
    rad_line_diff.set_data(rr, I_gauss(rr, sigma_det[1]))

    ax_rad.set_xlim(0, rr.max())
    ax_rad.set_ylim(-0.02, 1.02)

    fig.canvas.draw_idle()

# -------------------- Event handlers --------------------
def on_click(event):
    # only respond to clicks in the geometry subplot
    if event.inaxes != ax_geom:
        return
    if event.xdata is None or event.ydata is None:
        return

    state["x0"] = float(event.xdata)
    state["y0"] = float(event.ydata)
    update()

def on_slider(val):
    state["M"] = float(val)
    update()

fig.canvas.mpl_connect("button_press_event", on_click)
sM.on_changed(on_slider)

# Initial draw
update()
plt.show()
