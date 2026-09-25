import sys, os, time, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
for line in (HERE / ".env").read_text().splitlines():
    if line.startswith("POLYSIM_API_KEY="):
        os.environ["POLYSIM_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
log = open(HERE / "bot_out.log", "ab")
p = subprocess.Popen([sys.executable, str(HERE / "twap_bot.py"), "--mode", "paper", "--stake", "100"],
                     stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                     creationflags=0x00000208, close_fds=True)
print("LAUNCHED twap_bot.py pid", p.pid)
time.sleep(3)
print("alive", p.poll() is None)
log.close()
