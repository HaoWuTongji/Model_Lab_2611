"""在 Spyder 或任意 Python IDE 中直接运行以启动 ModelLab。"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def find_python() -> str:
    """优先使用项目虚拟环境中的 Python，便于在 Spyder 默认环境中直接运行。"""
    candidates = [
        ROOT / ".venv" / "bin" / "python",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return sys.executable


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_until_ready(url: str, processes: list[subprocess.Popen], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            return False
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def main() -> None:
    os.chdir(ROOT)
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise SystemExit("缺少 frontend/dist/index.html，请重新完整解压 ModelLab 安装包。")

    if is_port_in_use(HOST, PORT):
        try:
            with urllib.request.urlopen(f"{URL}/api/health", timeout=1.5) as resp:
                data = resp.read().decode("utf-8", errors="ignore")
                if resp.status == 200 and "database" in data:
                    print(f"检测到 ModelLab 已在运行 ({URL})，直接为你打开浏览器。", flush=True)
                    webbrowser.open(URL)
                    return
        except Exception:
            pass
        raise SystemExit(
            f"端口 {PORT} 已被其他程序占用，请先关闭占用 {PORT} 端口的程序后再启动 ModelLab。"
        )

    python_exe = find_python()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    backend_dir = str(ROOT / "backend")
    env["PYTHONPATH"] = (
        f"{backend_dir}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else backend_dir
    )

    processes: list[subprocess.Popen] = []

    def handle_sigterm(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_sigterm)

    print(f"正在使用 {python_exe} 启动 ModelLab...", flush=True)
    try:
        processes.append(
            subprocess.Popen(
                [
                    python_exe,
                    "-m",
                    "uvicorn",
                    "modellab.api:app",
                    "--host",
                    HOST,
                    "--port",
                    str(PORT),
                ],
                cwd=ROOT,
                env=env,
            )
        )
        processes.append(
            subprocess.Popen(
                [python_exe, "-m", "modellab.worker"],
                cwd=ROOT,
                env=env,
            )
        )

        if wait_until_ready(URL, processes):
            print(f"ModelLab 已就绪: {URL}", flush=True)
            print("在 Spyder 中点击停止按钮（或按 Ctrl+C）即可关闭服务。", flush=True)
            webbrowser.open(URL)
        elif all(p.poll() is None for p in processes):
            print(f"服务启动中，请稍候访问: {URL}", flush=True)

        while all(p.poll() is None for p in processes):
            time.sleep(0.5)
        for p in processes:
            if p.returncode:
                raise SystemExit(p.returncode)
    except KeyboardInterrupt:
        print("\n正在关闭 ModelLab...", flush=True)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        print("ModelLab 已停止。", flush=True)


if __name__ == "__main__":
    main()
