# reconnect_2400_test.py
# Quick connect/reconnect stress test for Keithley 2400 over GPIB.
# Prints full error queues at each phase so you can catch 803 / -420.

import time
import pyvisa

VISA_DLL = r"C:\Windows\System32\visa64.dll"   # change if needed, or set PYVISA_LIBRARY env var
RES      = "GPIB1::26::INSTR"                  # your 2400 address
CYCLES   = 5                                   # how many open/close cycles to test
DWELL_S  = 0.2                                 # small wait between steps

def drain_errors(inst, label=""):
    """Read :SYST:ERR? until '0' and print everything."""
    print(f"--- ERRORS {label} ---")
    for i in range(12):  # generous limit
        try:
            s = inst.query(":SYST:ERR?").strip()
        except Exception as e:
            print(f"  query failed on :SYST:ERR?: {e}")
            break
        print(f"  {i}: {s}")
        if s.startswith("0"):
            break

def init_2400(inst):
    """Minimal, 2400-safe init (no SAMP:COUN)."""
    cmds = [
        "*CLS",
        ":SOUR:FUNC VOLT",
        ":SOUR:VOLT 0",
        ":SENS:FUNC 'CURR'",
        ":SENS:CURR:PROT 0.01",   # 10 mA compliance
        ":FORM:ELEM VOLT,CURR",
        ":TRIG:COUN 1",           # (no :SAMP:COUN on classic 2400)
    ]
    for c in cmds:
        inst.write(c)
        time.sleep(0.02)

def do_one_cycle(rm, cycle_idx, turn_output_on=True):
    print(f"\n=== CYCLE {cycle_idx} ===")

    # Open
    inst = rm.open_resource(RES)
    inst.read_termination = "\n"
    inst.write_termination = "\n"
    inst.timeout = 5000

    # Immediately clear the device (GPIB viClear) to avoid stale I/O states
    try:
        inst.clear()
        print("viClear() OK")
    except Exception as e:
        print("viClear() failed:", e)

    # Clear and drain errors BEFORE any queries (prevents -420 / 803 on reconnect)
    inst.write("*CLS")
    drain_errors(inst, "after *CLS (pre-IDN)")

    # IDN (sanity check)
    try:
        idn = inst.query("*IDN?").strip()
        print("IDN:", idn)
    except Exception as e:
        print("IDN query failed:", e)

    # Set a known state
    init_2400(inst)
    drain_errors(inst, "after init")

    # Optional: set a small V and toggle output
    inst.write(":SOUR:VOLT 0.1")
    if turn_output_on:
        inst.write(":OUTP ON")
        time.sleep(DWELL_S)
        try:
            ans = inst.query("READ?").strip()
            print("READ? ->", ans)
        except Exception as e:
            print("READ? failed:", e)
        drain_errors(inst, "after READ?")
    else:
        inst.write(":OUTP OFF")
        drain_errors(inst, "after setting output OFF")

    # Abort any activity and close
    try:
        inst.write(":ABOR")
    except Exception:
        pass
    try:
        inst.close()
    except Exception as e:
        print("close failed:", e)

if __name__ == "__main__":
    rm = pyvisa.ResourceManager(VISA_DLL)
    # Alternate ON/OFF on each cycle so you can see both paths behave
    for k in range(1, CYCLES + 1):
        do_one_cycle(rm, k, turn_output_on=(k % 2 == 1))
        time.sleep(0.3)
    print("\nDone.")




## IN CASE OF ERROR -113

# # quick_test_2400_clean.py
# import pyvisa, time

# VISA_DLL = r"C:\Windows\System32\visa64.dll"
# RES = "GPIB1::26::INSTR"

# rm = pyvisa.ResourceManager(VISA_DLL)
# inst = rm.open_resource(RES)
# inst.read_termination = "\n"
# inst.write_termination = "\n"
# inst.timeout = 5000

# def err(): return inst.query(":SYST:ERR?").strip()

# # Drain old errors
# for _ in range(4):
#     if err().startswith("0"): break

# cmds = [
#     "*CLS",
#     ":SOUR:FUNC VOLT",
#     ":SOUR:VOLT 0",
#     ":SENS:FUNC 'CURR'",
#     ":SENS:CURR:PROT 0.01",
#     ":FORM:ELEM VOLT,CURR",
#     ":TRIG:COUN 1",
#     # (no :SAMP:COUN on classic 2400)
# ]

# print("Beginning 2400 clean init…\n")
# for c in cmds:
#     inst.write(c); time.sleep(0.05)
#     print(f"{c:<30} -> {err()}")

# print("\nSetting 0.1 V & reading once…")
# inst.write(":SOUR:VOLT 0.1")
# inst.write(":OUTP ON")
# time.sleep(0.1)
# print("READ? ->", inst.query("READ?").strip())
# print("Error after read ->", err())

# inst.write(":OUTP OFF")
# inst.close()



