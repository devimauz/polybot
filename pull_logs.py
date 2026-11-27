"""pull_logs.py - copy every price-log window from the AWS box into the local
price_logs/polymarket folder.  Never deletes anything locally.

Copies a file only when it is missing locally or the remote copy is larger
(a window still being written grows; a finished one stops).
"""
import os
import stat
import sys

sys.path.insert(0, r"C:\Users\user\Desktop\bot")
import _aws_ssh as A

REMOTE = "/home/ec2-user/poly/price_logs/polymarket"
LOCAL = r"C:\Users\user\Desktop\bot\price_logs\polymarket"


def main():
    os.makedirs(LOCAL, exist_ok=True)
    c = A.connect()
    s = c.open_sftp()
    s.chdir(REMOTE)
    names = [f.filename for f in s.listdir_attr()
             if f.filename.endswith((".jsonl", ".jsonl.gz"))]
    print("remote files:", len(names))
    got = skipped = 0
    total = 0
    for name in sorted(names):
        rp = REMOTE + "/" + name
        lp = os.path.join(LOCAL, name)
        rsz = s.stat(rp).st_size
        if os.path.exists(lp) and os.path.getsize(lp) >= rsz:
            skipped += 1
            continue
        s.get(rp, lp)
        got += 1
        total += rsz
        if got % 25 == 0:
            print("  pulled %d (%s)" % (got, name), flush=True)
    s.close()
    c.close()
    print("DONE pulled=%d already_present=%d bytes=%.1f MB" % (got, skipped, total / 1e6))


if __name__ == "__main__":
    main()
