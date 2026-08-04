import pyvisa, time
rm  = pyvisa.ResourceManager(r"C:\Windows\System32\visa64.dll")
res = "TCPIP0::10.164.14.244::5025::SOCKET"
inst = rm.open_resource(res)
inst.read_termination = "\n"; inst.write_termination = "\n"; inst.timeout = 5000

def err(): return inst.query(":SYST:ERR?").strip()
for _ in range(4):
    if err().startswith("0"): break

seq = [
    "*CLS",
    ":SOUR:FUNC VOLT",
    ":SOUR:VOLT 0",
    ":SOUR:VOLT:RANG:AUTO ON",
    ":SENS:VOLT:RANG:AUTO ON",
    ":SENS:CURR:RANG:AUTO ON",
    ":SOUR:VOLT:ILIM 0.01",
]

print("Beginning clean 2450 init test…\n")
for c in seq:
    inst.write(c); time.sleep(0.05)
    print(f"{c:<30} -> {err()}")

inst.close()
