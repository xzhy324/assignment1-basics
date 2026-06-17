import os
import resource

def log_mem(label: str):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[mem pid={os.getpid()}] {label}: maxrss={rss_kb / 1024:.1f} MiB", flush=True)