import os

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None

try:
    from rot.gui import config as rot_cfg
except Exception:
    rot_cfg = None

from measurements.config import DATA_DIR, ROOT_DIR

DEFAULT_STAGE_SERIAL = "27600915"
DEFAULT_STAGE_SCALE = "PRM1-Z8"

DEFAULT_EXPOSURE_MS = getattr(andor_cfg, "DEFAULT_EXPOSURE_MS", 50.0)
DEFAULT_ACQ_NUMBER = 1

DEFAULT_RAMP_STEP_DEG = getattr(rot_cfg, "DEFAULT_RAMP_STEP_DEG", 5.0)

DEFAULT_POWER_CALIB_PATH = os.path.join(ROOT_DIR, "test_power_2.csv")

DEFAULT_CROP_TOP = 0
DEFAULT_CROP_BOTTOM = 0
DEFAULT_CROP_LEFT = 0
DEFAULT_CROP_RIGHT = 0

DEFAULT_ROI_X1 = 470
DEFAULT_ROI_X2 = 575
DEFAULT_ROI_Y1 = 90
DEFAULT_ROI_Y2 = 150

POLL_MS = 500
