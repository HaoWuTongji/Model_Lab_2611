"""Trusted local bootstrap; executes only the dispatcher's generated template.

stdin is a lifetime pipe owned by the dispatcher. When it closes, terminate
the solver (and its private process group on POSIX).
This is process separation, not a container/security sandbox.
"""

import math
import os
import runpy
import signal
import sys
import threading

if os.name != "nt":
    import resource
else:
    import ctypes
    import msvcrt
    import time
    from ctypes import wintypes

    resource = None


def main():
    script, result, limit = sys.argv[1:]
    if resource is not None:
        cpu = max(1, math.ceil(float(limit) + 10))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def monitor_parent():
        if os.name == "nt":
            # A blocking stdin read can stall native solver imports on Windows.
            # The dispatcher never writes data. Poll until its write end closes.
            peek = ctypes.windll.kernel32.PeekNamedPipe
            peek.argtypes = (
                wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPDWORD,
            )
            peek.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(0)
            while peek(handle, None, 0, None, None, None):
                time.sleep(0.1)
            os._exit(1)
        while os.read(0, 1):
            pass
        os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=monitor_parent, daemon=True).start()
    sys.argv = [script, result]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
