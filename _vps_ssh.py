#!/usr/bin/env python3
"""SSH helper for the Amsterdam VPS (mirror of _aws_ssh.py).
Password lives in .vps_auth next to this file; key fallback: .vps_id_ed25519."""
import os, sys, paramiko

HOST = "45.136.212.21"
HERE = os.path.dirname(os.path.abspath(__file__))

def _client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    k = os.path.join(HERE, ".vps_id_ed25519")
    if os.path.exists(k):
        c.connect(HOST, username="root", pkey=paramiko.Ed25519Key.from_private_key_file(k), timeout=25)
    else:
        pw = open(os.path.join(HERE, ".vps_auth")).read().strip()
        c.connect(HOST, username="root", password=pw, timeout=25)
    return c

def run(cmd, timeout=300):
    c = _client()
    try:
        _, o, e = c.exec_command(cmd, timeout=timeout)
        return o.read().decode(errors="replace"), e.read().decode(errors="replace"), o.channel.exit_status
    finally:
        c.close()

def put(local, remote):
    c = _client()
    try:
        s = c.open_sftp()
        s.put(local, remote)
        s.close()
        print(f"put {local} -> {remote}")
    finally:
        c.close()

def get(remote, local):
    c = _client()
    try:
        s = c.open_sftp()
        s.get(remote, local)
        s.close()
        print(f"got {remote} -> {local}")
    finally:
        c.close()

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--put":
        put(sys.argv[2], sys.argv[3]); sys.exit(0)
    if len(sys.argv) >= 3 and sys.argv[1] == "--get":
        get(sys.argv[2], sys.argv[3]); sys.exit(0)
    out, err, rc = run(" ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo ok")
    print(out, end="")
    if err.strip(): print("STDERR:", err, file=sys.stderr)
    sys.exit(rc)
