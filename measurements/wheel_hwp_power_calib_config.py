from measurements.config import DATA_DIR  # noqa: F401  (re-exported for widget)

# Stage serials
DEFAULT_ND_SERIAL   = "27600911"
DEFAULT_HWP_SERIAL  = "27264008"
DEFAULT_STAGE_SCALE = "PRM1-Z8"

# Step 1 — HWP sweep: ND fixed at max-transmission angle, HWP swept
DEFAULT_ND_ANGLE       = 200.0   # ND fixed angle (max power) during HWP sweep
DEFAULT_HWP_START      = 0.0
DEFAULT_HWP_STOP       = 360.0
DEFAULT_HWP_STEP       = 0.5

# Step 2 — ND sweep: HWP fixed at max-power angle, ND swept
DEFAULT_HWP_FIXED_ANGLE = 0.0   # HWP fixed angle (max power) during ND sweep
DEFAULT_ND_START        = 0.0
DEFAULT_ND_STOP         = 360.0
DEFAULT_ND_STEP         = 5.0

# Shared
DEFAULT_N_READINGS    = 50    # power readings per step
DEFAULT_RAMP_STEP_DEG = 5.0   # motion ramp step size
DEFAULT_PM_AVERAGING  = 100   # TLPM hardware averaging count per reading
