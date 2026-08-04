# position_power_mapping.py
# Fit line1 (position_deg -> power_uW) on a circular 0–360 deg axis (unwrap across 360),
# then map line2 positions to power using the fitted curve, plot, and export CSV.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# -----------------------------
# Input data
# -----------------------------
line1 = """position_deg,power_uW
100,0.29
200.009704,0.75
250.009135,1.88
300.007524,4.82
350.011122,7.85
"""

line2 = """position_deg,intensity
100,1997513
105,1994617
110,2007647
115,2013466
120,2028922
125,2038193
130,2047832
135,2062193
140,2077142
145,2091004
150,2109290
155,2129244
160,2149483
165,2171312
170,2189659
175,2220310
180,2246190
185,2287170
190,2318294
195,2346556
200,2385224
205,2436680
210,2485696
215,2540531
220,2612565
225,2691965
230,2763699
235,2840300
240,2934580
245,3039868
250,3146685
255,3298741
260,3433790
265,3600051
270,3726942
275,3914915
280,4107340
285,4323448
290,4609810
295,4858982
300,5145233
305,5487928
310,5833918
315,6237220
320,6582345
325,6979960
330,7249468
"""

df1 = pd.read_csv(pd.io.common.StringIO(line1))
df2 = pd.read_csv(pd.io.common.StringIO(line2))

# -----------------------------
# Helpers
# -----------------------------
def unwrap_degrees(deg: np.ndarray) -> np.ndarray:
    """
    Unwrap degrees assuming the data progresses forward and may wrap at 360.
    Example: [350, 10] -> [350, 370]
    """
    deg = np.asarray(deg, dtype=float)
    out = deg.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1] - 180.0:
            out[i:] += 360.0
    return out

def unwrap_like_reference(deg: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    """
    Map degrees into the same "unwrapped band" as [ref_min, ref_max].
    For example, if ref range is ~[100, 370], then 0..10 becomes 360..370.
    """
    deg = np.asarray(deg, dtype=float)
    out = deg.copy()

    # First put everything into [0, 360] (handle 360 explicitly if present)
    out = np.mod(out, 360.0)
    # 360 should stay 360 not 0 if it exists in original
    out[np.isclose(deg, 360.0)] = 360.0

    # Then shift by +/-360 so points lie near the reference band
    # Strategy: if below ref_min by a lot, add 360; if above ref_max by a lot, subtract 360.
    out2 = out.copy()
    out2[out2 < ref_min - 90.0] += 360.0
    out2[out2 > ref_max + 90.0] -= 360.0
    return out2

# -----------------------------
# 1) Fit line1 with circular unwrap + PCHIP
# -----------------------------
x1_raw = df1["position_deg"].to_numpy(dtype=float)
y1 = df1["power_uW"].to_numpy(dtype=float)

x1 = unwrap_degrees(x1_raw)

# Sort just in case (PCHIP requires increasing x)
order1 = np.argsort(x1)
x1 = x1[order1]
y1 = y1[order1]

# Shape-preserving interpolator
fit_pos_to_power = PchipInterpolator(x1, y1, extrapolate=True)

ref_min, ref_max = float(np.min(x1)), float(np.max(x1))

# -----------------------------
# 2) Map line2 positions -> power using fitted curve
# -----------------------------
x2_raw = df2["position_deg"].to_numpy(dtype=float)
intensity = df2["intensity"].to_numpy(dtype=float)

x2_unwrapped = unwrap_like_reference(x2_raw, ref_min=ref_min, ref_max=ref_max)
power2 = fit_pos_to_power(x2_unwrapped)

df2_out = pd.DataFrame({
    "position_deg": x2_raw,
    "position_deg_unwrapped": x2_unwrapped,
    "power_uW_est": power2,
    "intensity": intensity,
})

# -----------------------------
# 3) Plot (line1 + mapped line2)
# -----------------------------
# For plotting, use an unwrapped x-axis so the wrap at 360 is not a discontinuity.
fig, ax1 = plt.subplots(figsize=(9, 5))

# line1 (points + fitted curve)
ax1.scatter(x1, y1, label="line1 data (unwrapped)", zorder=3)

xgrid = np.linspace(ref_min, ref_max, 600)
ax1.plot(xgrid, fit_pos_to_power(xgrid), label="line1 fit (PCHIP)")

# line2 mapped to power
# Plot as a curve in unwrapped coords to show continuity across 360
order2 = np.argsort(x2_unwrapped)
ax1.plot(x2_unwrapped[order2], power2[order2], label="line2 positions mapped to power")

ax1.set_xlabel("Position (deg, unwrapped)")
ax1.set_ylabel("Power (uW)")
ax1.grid(True, alpha=0.3)
ax1.legend()

fig.tight_layout()
fig.savefig("position_power_mapping.png", dpi=200)
plt.show()

# -----------------------------
# 4) Export CSV
# -----------------------------
# User-requested: "csv for position vs power in line 2"
df2_csv = df2_out[["position_deg", "power_uW_est"]].copy()
df2_csv.to_csv("line2_position_vs_power.csv", index=False)

print("Wrote: line2_position_vs_power.csv")
print("Wrote: position_power_mapping.png")
