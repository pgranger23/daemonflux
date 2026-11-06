"""Resume-safe production driver (milestone 4).

    python m4_driver.py NAME CAP_CORE_H NPROC_FN_FILE -- <sky3d.py arguments>

Runs ``sky3d.py`` in blocks of at most BLOCK_H hours (SIGTERM at the block end; tasks are files,
re-launching the same command resumes), long-tail strata first (SKY3D_TAIL_FIRST), and between
and during blocks logs to ``$M/logs/progress_NAME.txt``: done tasks, CPU (user+sys of the
finished tasks, = sum of the t-file ``meta[7]``) plus the in-flight estimate, free disk, RSS of
our jobs.  It stops (kills the job) if the CPU sum exceeds CAP, free disk < 150 GB, or total RSS
of our python processes > 35 GB.  Worker count = 20 - (workers of the other job), read from
the file given as 3rd argument (a number written by the other driver; 0 if absent).
"""
import glob, os, signal, subprocess, sys, time
import numpy as np

name, cap, other_file = sys.argv[1], float(sys.argv[2]), sys.argv[3]
cmd = sys.argv[sys.argv.index("--") + 1:]
M = os.environ["M"]; W = os.environ["W"]
BLOCK_H = float(os.environ.get("BLOCK_H", "3"))
MAXW = 20
tag = cmd[cmd.index("--tag") + 1]; outdir = cmd[cmd.index("--outdir") + 1]
tdir = os.path.join(outdir, f"{tag}_tasks")
prog = os.path.join(M, "logs", f"progress_{name}.txt")
mine = os.path.join(M, "work", f"workers_{name}.txt")
prior_f = os.path.join(M, "work", f"cpu_prior_{name}.txt")
prior = float(open(prior_f).read()) if os.path.exists(prior_f) else 0.0   # core-h of earlier driver runs


def log(s):
    with open(prog, "a") as f:
        f.write(time.strftime("%F %T ") + s + "\n")


def cpu_done():
    tot, n = 0.0, 0
    for p in glob.glob(os.path.join(tdir, "t*.npz")):
        try:
            tot += float(np.load(p, allow_pickle=False)["meta"][7]); n += 1
        except Exception:
            pass
    return tot / 3600.0, n


_cache = {}
def cpu_done_fast():
    tot, n = 0.0, 0
    for p in glob.glob(os.path.join(tdir, "t*.npz")):
        if p not in _cache:
            try:
                _cache[p] = float(np.load(p, allow_pickle=False)["meta"][7])
            except Exception:
                continue
        tot += _cache[p]; n += 1
    return tot / 3600.0, n


def _pids():
    out = subprocess.run(f"pgrep -u $USER -f 'sky3d.py.*--tag {tag}( |$)'", shell=True, capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def rss_gb():
    """Proportional set size (PSS: shared copy-on-write pages of the forked workers counted once) of our
    sky3d processes [GB].  (Summed RSS double counts the shared pages: it read 57 GB while `free` showed 14 GB used.)"""
    tot = 0
    for pid in _pids():
        try:
            for ln in open(f"/proc/{pid}/smaps_rollup"):
                if ln.startswith("Pss:"):
                    tot += int(ln.split()[1])
        except Exception:
            pass
    return tot / 1e6


_cpu_seen = {}
def cpu_live():
    """user+sys CPU [core-h] of all sky3d processes seen so far in this driver (max per pid, 100 Hz ticks)."""
    for pid in _pids():
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            _cpu_seen[pid] = max(_cpu_seen.get(pid, 0), int(f[11]) + int(f[12]))
        except Exception:
            pass
    return sum(_cpu_seen.values()) / 100.0 / 3600.0


def free_gb():
    st = os.statvfs(W)
    return st.f_bavail * st.f_frsize / 1e9


def other_workers():
    try:
        return int(open(other_file).read().split()[0])
    except Exception:
        return 0


def ntasks_total():
    m = os.path.join(tdir, "manifest.npz")
    return int(np.load(m)["ntask"][0]) if os.path.exists(m) else -1


env = dict(os.environ, SKY3D_TAIL_FIRST="1")
block = 0
while True:
    cpu0, n0 = cpu_done_fast()
    cpu0 = max(cpu0, prior)
    nt = ntasks_total()
    if nt > 0 and os.path.exists(os.path.join(outdir, f"{tag}_total.npz")):
        log(f"DONE: total file exists; CPU {cpu0:.2f} core-h, tasks {n0}/{nt}")
        break
    nproc = max(4, MAXW - other_workers())
    open(mine, "w").write(str(nproc))
    c = [a if a != "NPROC" else str(nproc) for a in cmd]
    block += 1
    log(f"block {block} start: nproc {nproc}, tasks done {n0}/{nt}, CPU so far {cpu0:.2f} core-h, free disk {free_gb():.0f} GB, RSS {rss_gb():.1f} GB")
    p = subprocess.Popen(c, env=env, stdout=open(os.path.join(M, "logs", f"{name}.log"), "a"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time(); stop = None
    while p.poll() is None:
        time.sleep(60)
        cpu, n = cpu_done_fast()
        live = prior + cpu_live()
        cpu = max(cpu, live)
        if time.time() - t0 > BLOCK_H * 3600:
            stop = "block time"
        elif cpu > cap:
            stop = f"CAP exceeded ({cpu:.1f} > {cap})"
        elif free_gb() < 150:
            stop = f"disk low ({free_gb():.0f} GB)"
        elif rss_gb() > 35:
            stop = f"RSS {rss_gb():.1f} GB"
        if int(time.time() - t0) % 600 < 60:
            log(f"  running: tasks done {n}/{nt}, CPU(finished tasks) {cpu:.2f} core-h, free disk {free_gb():.0f} GB, RSS {rss_gb():.1f} GB")
        if stop:
            log(f"  stopping the block: {stop}")
            os.killpg(p.pid, signal.SIGTERM); time.sleep(20)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
            p.wait()
            break
    cpu1, n1 = cpu_done()
    cpu1 = max(cpu1, prior + cpu_live())
    open(prior_f, "w").write(str(prior + cpu_live()))
    log(f"block {block} end (exit {p.returncode}{', ' + stop if stop else ''}): wall {(time.time()-t0)/3600:.2f} h, tasks done {n1}/{nt}, CPU {cpu1:.2f} core-h")
    if stop and stop != "block time":
        log("HALTED for the reason above; not relaunching"); break
    if stop is None and p.returncode not in (0,):
        log("sky3d exited with an error; halting"); break
open(mine, "w").write("0")
