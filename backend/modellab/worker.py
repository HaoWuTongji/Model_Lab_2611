"""Trusted dispatcher for explicit local subprocess or Docker execution."""

import concurrent.futures
import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
import sys

if os.name == "nt":
    import msvcrt
else:
    import fcntl
from functools import lru_cache
from pathlib import Path
from .storage import initialize, home, connect, transaction
from .compiler import compile_model, canonical, digest
from .exporter import write_bundle
from .validation import validate_result
from .providers import transcribe, diagnose, generate_from_text, chat_draft, analyze_result
from .spreadsheets import load_asset_tables

IMAGE = os.getenv("RUNNER_IMAGE", "modellab-solver:0.1")
STOP = threading.Event()


def docker_ready():
    if not shutil.which("docker"):
        return (
            False,
            "Docker 模式不可用：未安装 Docker。个人使用可设置 SOLVER_MODE=local。",
        )
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, timeout=5
        )
        return (
            (True, "")
            if r.returncode == 0
            else (False, "Docker 未启动或 solver-runner 镜像未构建，请运行 make runner")
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "Docker 服务不可用"


def solver_mode():
    return os.getenv("SOLVER_MODE", "local").strip().lower()


def local_environment(folder):
    tmp = Path(folder) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in (
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "PATHEXT",
                "USERPROFILE",
                "GUROBI_HOME",
                "GRB_LICENSE_FILE",
            )
        }
        env.update(
            {
                "TMP": str(tmp),
                "TEMP": str(tmp),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        return env
    return {
        "PATH": os.defpath,
        "LANG": "en_US.UTF-8",
        "TMPDIR": str(tmp),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _probe_env():
    if os.name == "nt":
        return {
            k: v
            for k, v in os.environ.items()
            if k.upper() in ("PATH", "SYSTEMROOT", "WINDIR", "GUROBI_HOME", "GRB_LICENSE_FILE")
        }
    return {"PATH": os.defpath}


@lru_cache(maxsize=1)
def local_ready():
    try:
        probe = subprocess.run(
            [sys.executable, "-I", "-c", "import highspy; highspy.Highs()"],
            env=_probe_env(),
            capture_output=True,
            timeout=10,
        )
        return (
            (True, "")
            if probe.returncode == 0
            else (False, "本机 HiGHS 加载失败，请运行 make setup-app 或安装依赖")
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "本机 Python/HiGHS 无法启动，请检查依赖安装"


@lru_cache(maxsize=1)
def available_solvers():
    out = {}
    for name, mod in [("highs", "highspy"), ("gurobi", "gurobipy"), ("cplex", "cplex")]:
        try:
            probe = subprocess.run(
                [sys.executable, "-I", "-c", f"import {mod}"],
                env=_probe_env(),
                capture_output=True,
                timeout=8,
            )
            out[name] = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            out[name] = False
    return out


def runner_ready(mode=None):
    mode = mode or solver_mode()
    if mode == "local":
        return local_ready()
    if mode == "docker":
        return docker_ready()
    return False, "SOLVER_MODE 仅支持 local 或 docker"


def launch_local(input_dir, output_dir, limit):
    kwargs = dict(
        cwd=input_dir,
        env=local_environment(output_dir.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("local_runner.py")),
            str(input_dir / "solve.py"),
            str(output_dir / "result.json"),
            str(limit),
        ],
        **kwargs,
    )


def stop_process(proc, jid, mode):
    if mode == "docker":
        remove(jid)
    if proc is None:
        return
    if mode == "local" and os.name != "nt":
        # proc owns a new session; never target a process group not created here.
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    elif proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if mode == "local" and os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        proc.wait(timeout=5)
    if proc.stdin:
        proc.stdin.close()


def monitor(proc, jid, mode, limit):
    deadline = time.monotonic() + limit + 10
    while proc.poll() is None:
        with transaction() as db:
            db.execute("UPDATE jobs SET heartbeat=? WHERE id=?", (time.time(), jid))
            cancelled = db.execute(
                "SELECT cancel FROM jobs WHERE id=?", (jid,)
            ).fetchone()[0]
        if cancelled or STOP.is_set():
            stop_process(proc, jid, mode)
            return "cancelled" if cancelled else "interrupted"
        if time.monotonic() > deadline:
            stop_process(proc, jid, mode)
            return "timeout"
        time.sleep(0.1)
    return "finished"


def command(jid, input_dir, output_dir, limit):
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        "modellab-" + jid,
        "--label",
        "modellab.job=" + jid,
        "--network",
        "none",
        "--user",
        "65532:65532",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        "1",
        "--memory",
        "1g",
        "--pids-limit",
        "64",
        "--ulimit",
        "fsize=10485760:10485760",
        "--log-driver",
        "none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        f"type=bind,src={input_dir},dst=/input,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        IMAGE,
        "python",
        "-I",
        "/input/solve.py",
        "/output/result.json",
    ]


def remove(jid):
    if shutil.which("docker"):
        try:
            subprocess.run(
                ["docker", "rm", "-f", "modellab-" + jid],
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def safe_result(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > 10 * 1024 * 1024:
            raise RuntimeError("求解结果文件无效或过大")
        raw = f.read(10 * 1024 * 1024 + 1)
    value = json.loads(
        raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite"))
    )
    allowed = {
        "termination",
        "has_incumbent",
        "proven_optimal",
        "objective_value",
        "best_bound",
        "relative_gap",
        "variables",
        "wall_time",
        "solver_time",
        "raw_status",
        "solver_version",
        "conflict_sources",
    }
    if set(value) - allowed or not {
        "termination",
        "has_incumbent",
        "proven_optimal",
        "variables",
        "objective_value",
    }.issubset(value):
        raise RuntimeError("求解结果 Schema 无效")
    if value["termination"] not in (
        "optimal",
        "infeasible",
        "unbounded",
        "infeasible_or_unbounded",
        "time_limit",
        "iteration_limit",
        "cancelled",
        "numerical_error",
        "input_error",
        "execution_error",
        "unknown",
    ):
        raise RuntimeError("未知求解状态")
    if (
        type(value["has_incumbent"]) is not bool
        or type(value["proven_optimal"]) is not bool
    ):
        raise RuntimeError("结果布尔字段无效")
    from .compiler import guard_tree

    guard_tree(value)
    from .result_schema import SolverResult

    return SolverResult.model_validate(value).model_dump(exclude_none=False)


def finish(jid, state, result=None, error=None):
    with transaction() as db:
        row = db.execute("SELECT cancel FROM jobs WHERE id=?", (jid,)).fetchone()
        if row and row["cancel"]:
            state = "cancelled"
            result = None
        db.execute(
            "UPDATE jobs SET state=?,result=?,error=?,finished=? WHERE id=?",
            (
                state,
                canonical(result) if result is not None else None,
                error,
                time.time(),
                jid,
            ),
        )


def process(j):
    jid = j["id"]
    proc = None
    mode = None
    thread = None
    try:
        with connect() as db:
            r = db.execute(
                "SELECT * FROM revisions WHERE id=?", (j["revision_id"],)
            ).fetchone()
        payload = json.loads(j["payload"])
        if j["kind"] in (
            "transcribe",
            "diagnose",
            "generate_draft",
            "chat_draft",
            "analyze_result",
        ):

            def cancelled():
                with connect() as db:
                    row = db.execute(
                        "SELECT cancel FROM jobs WHERE id=?", (jid,)
                    ).fetchone()
                return STOP.is_set() or bool(row["cancel"])

            if j["kind"] == "transcribe":
                result = transcribe(
                    json.loads(r["assets_json"]),
                    r["problem_text"],
                    payload["provider"],
                    cancelled,
                    connection_revision=payload.get("connection_revision"),
                )
            elif j["kind"] == "generate_draft":
                with connect() as db:
                    data_tables = load_asset_tables(json.loads(r["assets_json"]), db)
                result = generate_from_text(
                    r["problem_text"],
                    payload["provider"],
                    cancelled,
                    connection_revision=payload.get("connection_revision"),
                    data_tables=data_tables,
                )
                with transaction() as db:
                    row = db.execute(
                        "SELECT messages_json FROM ai_threads WHERE submission_id=?",
                        (r["submission_id"],),
                    ).fetchone()
                    msgs = json.loads(row["messages_json"]) if row else []
                    msgs.append(
                        dict(
                            id=j["id"],
                            role="assistant",
                            content=result["summary"],
                            updated_draft=True,
                            changes=["生成初始模型草稿"],
                            created=time.time(),
                        )
                    )
                    draft_obj = dict(
                        symbolic_model_draft=result["symbolic_model_draft"],
                        data=result.get("data") or {},
                        summary=result["summary"],
                        provider=result.get("provider"),
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO ai_threads(submission_id,draft_json,messages_json,updated) VALUES (?,?,?,?)",
                        (
                            r["submission_id"],
                            canonical(draft_obj),
                            canonical(msgs),
                            time.time(),
                        ),
                    )
            elif j["kind"] == "chat_draft":
                with connect() as db:
                    row = db.execute(
                        "SELECT draft_json,messages_json FROM ai_threads WHERE submission_id=?",
                        (r["submission_id"],),
                    ).fetchone()
                    data_tables = load_asset_tables(json.loads(r["assets_json"]), db)
                existing_draft = (
                    json.loads(row["draft_json"])
                    if row and row["draft_json"]
                    else None
                )
                history = json.loads(row["messages_json"]) if row else []
                # Exclude the just-appended current user message from prior history
                prior = history[:-1] if history and history[-1].get("role") == "user" else history
                base_model = (
                    existing_draft["symbolic_model_draft"]
                    if existing_draft
                    else json.loads(r["model_json"])
                )
                base_data = (
                    existing_draft.get("data", {})
                    if existing_draft
                    else json.loads(r["data_json"])
                )
                result = chat_draft(
                    r["problem_text"],
                    base_model,
                    base_data,
                    prior,
                    payload["message"],
                    payload["provider"],
                    cancelled,
                    connection_revision=payload.get("connection_revision"),
                    data_tables=data_tables,
                )
                with transaction() as db:
                    row2 = db.execute(
                        "SELECT draft_json,messages_json FROM ai_threads WHERE submission_id=?",
                        (r["submission_id"],),
                    ).fetchone()
                    msgs = json.loads(row2["messages_json"]) if row2 else []
                    msgs.append(
                        dict(
                            id=j["id"],
                            role="assistant",
                            content=result["reply"],
                            updated_draft=bool(result["updated"]),
                            changes=result.get("changes", []),
                            created=time.time(),
                        )
                    )
                    next_draft = (
                        json.loads(row2["draft_json"])
                        if row2 and row2["draft_json"]
                        else None
                    )
                    if result["updated"] and result.get("symbolic_model_draft") is not None:
                        next_draft = dict(
                            symbolic_model_draft=result["symbolic_model_draft"],
                            data=result["data"]
                            if result.get("data") is not None
                            else base_data,
                            summary=result["reply"],
                            provider=result.get("provider"),
                        )
                    db.execute(
                        "INSERT OR REPLACE INTO ai_threads(submission_id,draft_json,messages_json,updated) VALUES (?,?,?,?)",
                        (
                            r["submission_id"],
                            canonical(next_draft) if next_draft is not None else None,
                            canonical(msgs),
                            time.time(),
                        ),
                    )
            elif j["kind"] == "analyze_result":
                with connect() as db:
                    sj = db.execute(
                        "SELECT result FROM jobs WHERE id=?",
                        (payload["solve_job_id"],),
                    ).fetchone()
                if not sj or not sj["result"]:
                    raise RuntimeError("未找到对应的求解结果")
                result = analyze_result(
                    r["problem_text"],
                    json.loads(r["model_json"]),
                    json.loads(r["data_json"]),
                    json.loads(sj["result"]),
                    payload.get("user_request", ""),
                    payload["provider"],
                    cancelled,
                    connection_revision=payload.get("connection_revision"),
                )
                result["solve_job_id"] = payload["solve_job_id"]
                result["user_request"] = payload.get("user_request", "")
            else:
                result = diagnose(
                    json.loads(r["model_json"]),
                    json.loads(r["data_json"]),
                    r["problem_text"],
                    payload["provider"],
                    cancelled,
                    connection_revision=payload.get("connection_revision"),
                )
            finish(jid, "succeeded", result)
            return
        from .api import semantic_hash

        if j["confirmation_hash"] != semantic_hash(r):
            raise RuntimeError("模型确认 hash 不匹配")
        mode = payload.get("execution_mode", solver_mode())
        ready, reason = runner_ready(mode)
        if not ready:
            raise RuntimeError(reason)
        ir = compile_model(
            json.loads(r["model_json"]), json.loads(r["data_json"]), r["id"]
        )
        folder = home() / "jobs" / jid
        inp = folder / "input"
        out = folder / "output"
        out.mkdir(parents=True, exist_ok=False)
        out.chmod(0o777 if mode == "docker" else 0o700)
        solver_name = payload.get("solver", "highs")
        if mode == "local" and not available_solvers().get(solver_name, False):
            raise RuntimeError(
                f"当前 Python 环境未安装 {solver_name.upper()}，请先安装对应求解器或改用 HiGHS"
            )
        provenance = write_bundle(
            ir,
            inp,
            time_limit=payload["time_limit"],
            solver=solver_name,
            mip_gap=payload.get("mip_gap"),
            threads=payload.get("threads", 1),
            seed=payload.get("seed", 0),
            presolve=payload.get("presolve", "auto"),
        )
        if mode == "local":
            proc = launch_local(inp, out, payload["time_limit"])
        else:
            proc = subprocess.Popen(
                command(jid, inp, out, payload["time_limit"]),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        # Drain continuously into a bounded memory buffer; Docker writes no unbounded daemon log.
        logs = bytearray()

        def drain():
            while chunk := proc.stdout.read(8192):
                if len(logs) < 2 * 1024 * 1024:
                    logs.extend(chunk[: 2 * 1024 * 1024 - len(logs)])

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        outcome = monitor(proc, jid, mode, payload["time_limit"])
        if outcome != "finished":
            if outcome == "timeout":
                finish(jid, "failed", error="硬超时终止，未取得完整结果")
            else:
                finish(jid, outcome)
            return
        thread.join(timeout=2)
        if proc.returncode:
            raise RuntimeError("求解子进程失败，退出码 " + str(proc.returncode))
        result = safe_result(out / "result.json")
        result["validation"] = validate_result(ir, result)
        result.update(
            provenance=provenance,
            execution_mode=mode,
            log=logs.decode(errors="replace"),
            size=dict(
                variables=len(ir["variables"]), constraints=len(ir["constraints"])
            ),
        )
        if result["proven_optimal"] and not result["validation"]["passed"]:
            result["proven_optimal"] = False
            result["warning"] = "求解器报告最优，但平台校验未通过"
        finish(jid, "succeeded", result)
    except Exception as exc:
        # Do not disclose paths, credentials, HTTP request bodies or exception reprs.
        message = (
            str(exc)
            if isinstance(exc, RuntimeError)
            else "任务失败：" + type(exc).__name__
        )
        finish(jid, "failed", error=message)
    finally:
        if j["kind"] == "solve":
            stop_process(proc, jid, mode)
        if thread:
            thread.join(timeout=3)
        if proc and proc.stdout:
            proc.stdout.close()


def recover():
    with transaction() as db:
        rows = db.execute(
            "SELECT id,kind,payload FROM jobs WHERE state='running'"
        ).fetchall()
        for j in rows:
            if (
                j["kind"] == "solve"
                and json.loads(j["payload"]).get("execution_mode", "docker") == "docker"
            ):
                remove(j["id"])
        db.execute(
            "UPDATE jobs SET state='interrupted',error='worker 重启中断；请手动重新提交',finished=? WHERE state='running'",
            (time.time(),),
        )


def claim(kind):
    with transaction() as db:
        j = db.execute(
            "SELECT * FROM jobs WHERE state='queued' AND cancel=0 AND kind=? ORDER BY created LIMIT 1",
            (kind,),
        ).fetchone()
        if not j:
            return None
        db.execute(
            "UPDATE jobs SET state='running',started=?,heartbeat=? WHERE id=?",
            (time.time(), time.time(), j["id"]),
        )
        return dict(j)


def main():
    initialize()
    lock = open(home() / "worker.lock", "a")
    try:
        if os.name == "nt":
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        raise SystemExit("已有 worker 运行")
    recover()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    futures = {}
    concurrency = min(2, max(1, int(os.getenv("QUEUE_CONCURRENCY", "2"))))
    last = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency + 1) as pool:
        while not STOP.is_set():
            if time.monotonic() - last > 3:
                ready, reason = runner_ready()
                with transaction() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO runtime VALUES (?,?,?)",
                        (
                            "worker",
                            canonical(
                                dict(
                                    runner=ready,
                                    reason=reason,
                                    runner_mode=solver_mode(),
                                    solvers=available_solvers()
                                    if solver_mode() == "local"
                                    else {"highs": ready, "gurobi": False, "cplex": False},
                                )
                            ),
                            time.time(),
                        ),
                    )
                last = time.monotonic()
            futures = {f: k for f, k in futures.items() if not f.done()}
            for kind, max_count in [("solve", concurrency), ("transcribe", 1), ("diagnose", 1), ("generate_draft", 1), ("chat_draft", 1), ("analyze_result", 1)]:
                if sum(k == kind for k in futures.values()) < max_count:
                    j = claim(kind)
                    if j:
                        futures[pool.submit(process, j)] = kind
            STOP.wait(0.3)
    with transaction() as db:
        db.execute("DELETE FROM runtime WHERE key='worker'")


if __name__ == "__main__":
    main()
