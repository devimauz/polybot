"""SSH helper for the Dublin VPS. Password auth from .dub_auth."""
import sys, os, paramiko

HOST = "72.11.157.52"
HERE = os.path.dirname(os.path.abspath(__file__))

def cli():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    pw = open(os.path.join(HERE, ".dub_auth")).read().strip()
    c.connect(HOST, username="root", password=pw, timeout=20,
              look_for_keys=False, allow_agent=False)
    return c

def main():
    c = cli()
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--put":
            sftp = c.open_sftp()
            sftp.put(sys.argv[2], sys.argv[3])
            print("put", sys.argv[2], "->", sys.argv[3])
            return
        if len(sys.argv) > 1 and sys.argv[1] == "--get":
            sftp = c.open_sftp()
            sftp.get(sys.argv[2], sys.argv[3])
            print("get", sys.argv[2], "->", sys.argv[3])
            return
        cmd = " ".join(sys.argv[1:])
        _, o, e = c.exec_command(cmd, timeout=280, get_pty=False)
        out = o.read().decode("utf-8", "replace")
        err = e.read().decode("utf-8", "replace")
        print(out)
        if err.strip():
            sys.stderr.write("STDERR:\n" + err + "\n")
        o.channel.recv_exit_status()
    finally:
        c.close()

if __name__ == "__main__":
    main()
