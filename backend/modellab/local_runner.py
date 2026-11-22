"""Trusted local bootstrap; executes only the dispatcher's generated template.

stdin is a lifetime pipe owned by the dispatcher. EOF means the dispatcher died;
kill our entire private process group so a worker crash cannot leave a solver.
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
    resource = None


def main():
    script, result, limit = sys.argv[1:]
    if resource is not None:
        cpu = max(1, math.ceil(float(limit) + 10))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def monitor_parent():
        while os.read(0, 1):
            pass
        if os.name == "nt":
            os._exit(1)
        else:
            os.killpg(os.getpgrp(), signal.SIGKILL)

    threading.Thread(target=monitor_parent, daemon=True).start()
    sys.argv = [script, result]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
