# FIND KEITHLEY VIA LAN, GPIB & SERIAL.
# Compatible for Keithley 2450 & 2400
# Minxue 10.1.2025
#
# Scans:
#   - LAN (fixed IPs) for a Keithley 2450  → VXI-11 (inst0) and SCPI socket (5025)
#   - Serial (ASRL) for Keithley 2400/others (tries common bauds/terminations)
#   - GPIB instruments (enumerate; fallback brute force on GPIB0 addrs 1..30)
#
# Prints ready-to-use VISA resource strings.

import warnings
import pyvisa
from pyvisa import constants as pvconst

# ---------- CONFIG (edit as needed) ----------
LAN_IPS     = ["169.254.153.76", "10.164.14.233"]   # <-- put both IPs here (as many as you want)
VISA_DLL    = r"C:\Windows\System32\visa64.dll"     # Path to NI-VISA (adjust if needed)
SERIAL_BAUDS = [115200, 57600, 38400, 19200, 9600]
DEFAULT_TIMEOUT_MS = 4000
BRUTE_GPIB_IFACE = "GPIB0"                          # Interface to brute-force if none enumerate
BRUTE_GPIB_ADDRS = list(range(1, 31))               # 1..30 typical; change if you know the address
# ---------------------------------------------

def open_rm():
    return pyvisa.ResourceManager(VISA_DLL)

def safe_set(inst, attr, value):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(inst, attr, value)
    except Exception:
        pass

def try_idn(inst):
    safe_set(inst, 'timeout', DEFAULT_TIMEOUT_MS)
    # Try a few terminations and also write/read fallback
    for wt in ('\n', '\r\n'):
        for rt in ('\n', '\r\n'):
            try:
                inst.write_termination = wt
                inst.read_termination  = rt
                return inst.query("*IDN?").strip()
            except Exception:
                try:
                    inst.write("*IDN?")
                    s = inst.read().strip()
                    if s:
                        return s
                except Exception:
                    pass
    raise RuntimeError("No response to *IDN?")

def probe(rm, resource):
    try:
        inst = rm.open_resource(resource)
        idn = try_idn(inst)
        inst.close()
        return (resource, idn, "OK")
    except Exception as e:
        return (resource, None, f"{e}")

def scan_lan(rm):
    print("\n--- LAN (2450) ---")
    results = []
    for ip in LAN_IPS:
        resources = [
            f"TCPIP0::{ip}::inst0::INSTR",   # VXI-11
            f"TCPIP0::{ip}::5025::SOCKET",   # SCPI socket (5025)
        ]
        for r in resources:
            results.append(probe(rm, r))
    return results

def scan_serial(rm):
    print("\n--- Serial (ASRL) ---")
    results = []
    try:
        all_res = list(rm.list_resources())
    except Exception:
        all_res = []
    serials = [r for r in all_res if r.upper().startswith("ASRL")]
    for res in serials:
        try:
            inst = rm.open_resource(res)
        except Exception as e:
            results.append((res, None, f"open failed: {e}"))
            continue
        ok = False
        for baud in SERIAL_BAUDS:
            try:
                safe_set(inst, 'baud_rate', baud)
                safe_set(inst, 'data_bits', 8)
                safe_set(inst, 'parity', pvconst.Parity.none)
                safe_set(inst, 'stop_bits', pvconst.StopBits.one)
                safe_set(inst, 'flow_control', pvconst.VI_ASRL_FLOW_NONE)
                idn = try_idn(inst)
                results.append((f"{res} (baud={baud})", idn, "OK"))
                ok = True
                break
            except Exception:
                continue
        if not ok:
            results.append((res, None, "no response at common bauds"))
        try:
            inst.close()
        except Exception:
            pass
    return results

def scan_gpib(rm):
    print("\n--- GPIB ---")
    results = []
    # First: enumerate any GPIB resources VISA already sees
    try:
        all_res = list(rm.list_resources())
    except Exception:
        all_res = []
    gpibs = [r for r in all_res if r.upper().startswith("GPIB")]
    if gpibs:
        for r in gpibs:
            results.append(probe(rm, r))
        return results

    # Fallback: brute-force typical primary addresses on BRUTE_GPIB_IFACE
    print(f"(No GPIB resources enumerated; brute-forcing {BRUTE_GPIB_IFACE} addresses {BRUTE_GPIB_ADDRS[0]}..{BRUTE_GPIB_ADDRS[-1]})")
    for addr in BRUTE_GPIB_ADDRS:
        r = f"{BRUTE_GPIB_IFACE}::{addr}::INSTR"
        results.append(probe(rm, r))
    return results

def main():
    rm = open_rm()
    print("Using VISA:", rm.visalib)

    results = []
    results += scan_lan(rm)
    # results += scan_serial(rm)
    results += scan_gpib(rm)

    # Print summary
    print("\n=== RESULTS ===")
    good, bad = [], []
    for res, idn, note in results:
        if idn:
            good.append((res, idn, note))
        else:
            bad.append((res, idn, note))

    def is_keithley(s): return s and ("KEITHLEY" in s.upper())

    any_keithley = False
    for res, idn, note in good:
        tag = " [KEITHLEY]" if is_keithley(idn) else ""
        print(f"[OK]{tag} {res:40s}  {note:6s}  IDN={idn}")
        any_keithley = any_keithley or is_keithley(idn)

    if bad:
        print("\n--- Non-responders / errors ---")
        for res, _, note in bad:
            print(f"[--] {res:40s}  {note}")

    if any_keithley:
        print("\n=== RECOMMENDED (copy these into your app) ===")
        for res, idn, _ in good:
            if is_keithley(idn):
                print(res)
    else:
        print("\n(No Keithley instruments responded.)")

if __name__ == "__main__":
    main()
