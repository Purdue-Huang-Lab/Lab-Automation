import os

try:
    from andor.gui import config as andor_cfg
except Exception:
    andor_cfg = None

try:
    from rot.gui import config as rot_cfg
except Exception:
    rot_cfg = None


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Directory where measurement data is saved. Override by setting the
# AUTOMATION_DATA_DIR environment variable, e.g. in a .env file or shell profile.
DATA_DIR = os.environ.get(
    "AUTOMATION_DATA_DIR",
    r"TMDC Gated\260402gatedtrilayer",
)

DEFAULT_STAGE_SERIAL = "27600825"
DEFAULT_STAGE_B_SERIAL = "27600915"
DEFAULT_STAGE_SCALE = "PRM1-Z8"

DEFAULT_EXPOSURE_MS = 500.0
DEFAULT_ACQ_NUMBER = 1
DEFAULT_READOUT_RATE = "2.6MHz at 16-bit"
DEFAULT_PREAMP_GAIN = "4x"
DEFAULT_OUTPUT_AMP = "Conventional"

DEFAULT_GRATING = 1
DEFAULT_CENTER_WL_NM = 850.0
DEFAULT_SLIT_UM = 100.0
DEFAULT_SPEC_INDEX = 0
DEFAULT_SLIT_ID = "input_side"

if andor_cfg is not None:
    DEFAULT_READOUT_RATE = getattr(andor_cfg, "DEFAULT_READOUT_RATE", DEFAULT_READOUT_RATE)
    DEFAULT_OUTPUT_AMP = getattr(andor_cfg, "DEFAULT_OUTPUT_AMP", DEFAULT_OUTPUT_AMP)
    DEFAULT_SPEC_INDEX = getattr(andor_cfg, "DEFAULT_SPEC_INDEX", DEFAULT_SPEC_INDEX)
    DEFAULT_SLIT_ID = getattr(andor_cfg, "DEFAULT_SLIT_ID", DEFAULT_SLIT_ID)

DEFAULT_RAMP_STEP_DEG = getattr(rot_cfg, "DEFAULT_RAMP_STEP_DEG", 5.0)
DEFAULT_STAGE_ACCEL = getattr(rot_cfg, "DEFAULT_ACCEL", 10.0)

DEFAULT_POWER_CALIB_PATH = os.path.join(ROOT_DIR, "test_power_2.csv")

DEFAULT_CROP_TOP = 50
DEFAULT_CROP_BOTTOM = 50
DEFAULT_CROP_LEFT = 500
DEFAULT_CROP_RIGHT = 200
DEFAULT_LINECUT_ROW = 120
DEFAULT_LINECUT_WIDTH = 1
DEFAULT_ROI_X1 = 470
DEFAULT_ROI_X2 = 575
DEFAULT_ROI_Y1 = 90
DEFAULT_ROI_Y2 = 150

POLL_MS = 500
