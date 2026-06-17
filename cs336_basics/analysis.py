import os
import resource
import time

def log_mem(label: str):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print(f"[{timestamp} mem pid={os.getpid()}] {label}: maxrss={rss_kb / 1024:.1f} MiB", flush=True)