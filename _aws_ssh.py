"""_aws_ssh.py - run commands on the AWS EC2 logging box (Bitvise profile / aws.tlp).

The host, user and key come from the machine's own Bitvise setup:
  C:\\Users\\user\\Documents\\aws.tlp   (host + user)
  C:\\Users\\user\\Downloads\\home.ppk  (unencrypted PuTTY key)

Usage:
    python _aws_ssh.py "uptime; ls ~/poly"
    python _aws_ssh.py --file local.txt          # run a whole script remotely
    python _aws_ssh.py --put local remote        # scp-style upload
    python _aws_ssh.py --get remote local
"""
import base64
import os
import re
import struct
import sys

import paramiko

PROFILE = r"C:\Users\user\Documents\aws.tlp"
PPK = r"C:\Users\user\Downloads\home.ppk"


def _rd_mpint(b, o):
    (ln,) = struct.unpack(">I", b[o:o + 4])
    o += 4
    return int.from_bytes(b[o:o + ln], "big"), o + ln


def load_key(path=PPK):
    hdr, pub, priv, mode = {}, [], [], None
    for line in open(path, "rb").read().decode("ascii", "ignore").splitlines():
        if line.startswith("Public-Lines:"):
            mode = "pub"; continue
        if line.startswith("Private-Lines:"):
            mode = "priv"; continue
        if line.startswith("Private-MAC:"):
            mode = None; continue
        if mode is None and ":" in line and not line.startswith("AAAA"):
            k, v = line.split(":", 1)
            hdr[k.strip()] = v.strip()
            continue
        if mode == "pub":
            pub.append(line.strip())
        elif mode == "priv":
            priv.append(line.strip())
    pb = base64.b64decode("".join(pub))
    vb = base64.b64decode("".join(priv))
    (tlen,) = struct.unpack(">I", pb[:4])
    e, o = _rd_mpint(pb, 4 + tlen)
    n, o = _rd_mpint(pb, o)
    d, o2 = _rd_mpint(vb, 0)
    p, o2 = _rd_mpint(vb, o2)
    q, o2 = _rd_mpint(vb, o2)
    from cryptography.hazmat.primitives.asymmetric import rsa
    sk = rsa.RSAPrivateNumbers(
        p=p, q=q, d=d, dmp1=d % (p - 1), dmq1=d % (q - 1), iqmp=pow(q, -1, p),
        public_numbers=rsa.RSAPublicNumbers(e=e, n=n)).private_key()
    return paramiko.RSAKey(key=sk)


def profile(path=PROFILE):
    data = open(path, "rb").read()
    strs = re.findall(rb"[\x20-\x7e]{4,}", data)
    host = next(s.decode() for s in strs if re.fullmatch(rb"\d+\.\d+\.\d+\.\d+", s))
    user = next(s.decode() for s in strs
                if s in (b"ec2-user", b"ubuntu", b"root", b"admin", b"debian"))
    return host, user


def connect():
    host, user = profile()
    # While the WireGuard VPN is active, direct TCP-22 to the public IP is
    # blocked locally (WinError 10013); reach sshd through the tunnel instead.
    last = None
    for h in (host, "10.8.0.1"):
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            c.connect(h, username=user, pkey=load_key(), timeout=15,
                      allow_agent=False, look_for_keys=False)
            return c
        except Exception as e:
            c.close()
            last = e
    raise last


def run(cmd, timeout=180):
    c = connect()
    try:
        _, out, err = c.exec_command(cmd, timeout=timeout)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        return o, e, out.channel.recv_exit_status()
    finally:
        c.close()


def main(argv):
    if not argv:
        print(__doc__)
        return
    if argv[0] == "--file":
        cmd = open(argv[1], "rb").read().decode()
    elif argv[0] == "--put":
        c = connect()
        s = c.open_sftp()
        s.put(argv[1], argv[2])
        print("put", argv[1], "->", argv[2])
        s.close(); c.close(); return
    elif argv[0] == "--get":
        c = connect()
        s = c.open_sftp()
        s.get(argv[1], argv[2])
        print("got", argv[1], "->", argv[2])
        s.close(); c.close(); return
    else:
        cmd = " ".join(argv)
    o, e, rc = run(cmd)
    sys.stdout.write(o)
    if e.strip():
        sys.stdout.write("\n[stderr]\n" + e)
    print("\n[exit %d]" % rc)


if __name__ == "__main__":
    main(sys.argv[1:])
