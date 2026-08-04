VISA_DLL = r"C:\Windows\System32\visa64.dll"  # set "" for system default

DEFAULT_FRONT_RESOURCE = "GPIB0::24::INSTR"
DEFAULT_BACK_RESOURCE = "GPIB0::23::INSTR"

# Gate polling (when idle, not sweeping)
GATE_POLL_MS = 500

# Software over-current defaults
OC_LIMIT_A = 0.01       # 10 mA
OC_TRIP_SAMPLES = 2

# Gate ramp micro-step settings
RAMP_STEP_V = 0.01      # V per micro-step
RAMP_DWELL_S = 0.10     # seconds per micro-step
