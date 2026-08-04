import os

# ---- Device defaults ----
VISA_DLL           = r"C:\Windows\System32\visa64.dll"
DEFAULT_DLL_PATH   = r"C:\Program Files\PicoQuant\PH300-PHLibv30\demos\64\c\TTTRmode\PHLib64.dll"
DEFAULT_GATE_A_RESOURCE = "GPIB0::24::INSTR"
DEFAULT_GATE_B_RESOURCE = "GPIB0::23::INSTR"
DEFAULT_ROT_SERIAL = "27600911"

# ---- PH300 defaults ----
DEFAULT_TARGET_BIN_PS  = 8.0
DEFAULT_TACQ_MS        = 5000
DEFAULT_SYNC_DIV       = 1
DEFAULT_SYNC_OFFSET_PS = 30000
DEFAULT_HIST_OFFSET_PS = 0
DEFAULT_CH0_LEVEL      = 100
DEFAULT_CH0_ZC         = 20
DEFAULT_CH1_LEVEL      = 100
DEFAULT_CH1_ZC         = 20

# ---- Gate defaults ----
DEFAULT_ICOMP_NA   = 10.0       # nA
DEFAULT_GATE_V     = 0.0

# ---- Settle / timing ----
DEFAULT_GATE_SETTLE_S  = 0.5
DEFAULT_WHEEL_SETTLE_S = 1.0
RAMP_STEP_V  = 0.01
RAMP_DWELL_S = 0.10
POLL_MS      = 500

# ---- Plot defaults ----
DEFAULT_XMIN_PS = 60000.0
DEFAULT_XMAX_PS = 100000.0

# ---- Misc ----
BTN_W = 110
