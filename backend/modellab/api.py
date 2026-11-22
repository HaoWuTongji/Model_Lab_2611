from __future__ import annotations
import copy
import hashlib
import io
import json
import os
import secrets
import sqlite3
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, Request, Response, Depends, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError, Field
from PIL import Image, UnidentifiedImageError
from .schema import Strict, ModelDoc
from .storage import initialize, connect, transaction, home, uid, PROJECT
from .compiler import ModelError, compile_model, canonical, digest, guard_tree
from .parser import parse_expression
from .exporter import generate, generate_lp, model_stats
from .spreadsheets import parse_spreadsheet_bytes
from .settings import (
    ConnectionInput,
    public_connection,
    save_connection,
    clear_connection,
)
from .examples import CATALOG, example
from starlette.middleware.trustedhost import TrustedHostMiddleware


class APIError(Exception):
    def __init__(self, status, code, message, details=None):
        self.status, self.code, self.message, self.details = (
            status,
            code,
            message,
            details,
        )


def audit(db, sid, actor, event, payload):
    db.execute(
        "INSERT INTO audit VALUES (?,?,?,?,?,?)",
        (uid(), sid, actor, event, canonical(payload), time.time()),
    )


def get_submission(db, sid, user, write=False):
    row = db.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
    if not row:
        raise APIError(404, "NOT_FOUND", "未找到模型项目")
    return row


def get_revision(db, rid, user, write=False, current=False):
    r = db.execute("SELECT * FROM revisions WHERE id=?", (rid,)).fetchone()
    if not r:
        raise APIError(404, "NOT_FOUND", "未找到版本")
    s = get_submission(db, r["submission_id"], user, write)
    if current and s["current_revision"] != rid:
        raise APIError(409, "STALE_REVISION", "历史版本不能发起新操作，请复制为新版本")
    return r, s


def revision_json(r):
    value = dict(r)
    for name in ("model", "data", "assets"):
        value[name] = json.loads(value.pop(name + "_json"))
    return value


def semantic_hash(r):
    return digest(
        dict(
            model=json.loads(r["model_json"]),
            data=json.loads(r["data_json"]),
            problem=r["problem_text"],
            assets=json.loads(r["assets_json"]),
        )
    )


def confirmed(db, r):
    c = db.execute(
        "SELECT * FROM confirmations WHERE revision_id=?", (r["id"],)
    ).fetchone()
    if not c or c["hash"] != semantic_hash(r):
        raise APIError(409, "CONFIRMATION_REQUIRED", "请先确认当前模型、变量范围与数据")
    return c


def add_revision(db, s, user, body, source="manual"):
    if body.expected_version != s["version"]:
        raise APIError(409, "STALE_REVISION", "模型已被更新，请重新载入后编辑")
    guard_tree(body.model)
    guard_tree(body.data)
    # Drafts may omit bounds but must remain renderable. Do not persist the validation placeholders.
    shape = copy.deepcopy(body.model)
    for variable in shape.get("variables", []):
        if isinstance(variable, dict):
            variable.setdefault("lower", None)
            variable.setdefault("upper", None)
    normalized = ModelDoc.model_validate(shape).model_dump()
    for old, new in zip(body.model.get("variables", []), normalized["variables"]):
        for key in ("lower", "upper"):
            if key not in old:
                new.pop(key, None)
    body.model = normalized
    if len(body.assets) > 4:
        raise APIError(422, "ASSET_LIMIT", "每个版本最多四张图片")
    seen = set()
    for a in body.assets:
        if (
            set(a) != {"id", "rotation"}
            or a["rotation"] not in (0, 90, 180, 270)
            or a["id"] in seen
        ):
            raise APIError(422, "ASSET_LAYOUT", "图片调序或旋转信息无效")
        seen.add(a["id"])
        found = db.execute(
            "SELECT id FROM assets WHERE id=? AND submission_id=?", (a["id"], s["id"])
        ).fetchone()
        if not found:
            raise APIError(422, "ASSET_OWNER", "图片不属于该作业")
    rid = uid()
    num = s["version"] + 1
    db.execute(
        "INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            rid,
            s["id"],
            num,
            s["current_revision"],
            body.problem_text,
            canonical(body.model),
            canonical(body.data),
            canonical(body.assets),
            source,
            time.time(),
        ),
    )
    db.execute(
        "UPDATE submissions SET current_revision=?,version=? WHERE id=?",
        (rid, num, s["id"]),
    )
    audit(
        db,
        s["id"],
        user["id"],
        "revision_created",
        dict(
            revision_id=rid,
            parent_id=s["current_revision"],
            source=source,
            model_hash=digest(body.model),
            data_hash=digest(body.data),
        ),
    )
    return dict(id=rid, version=num)


@asynccontextmanager
async def lifespan(app):
    initialize()
    yield


app = FastAPI(title="ModelLab", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv(
        "MODELLAB_HOSTS", "127.0.0.1,localhost,::1,testserver"
    ).split(","),
)


@app.exception_handler(APIError)
async def api_error(request, exc):
    return JSONResponse(
        dict(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=getattr(request.state, "request_id", ""),
        ),
        status_code=exc.status,
    )


@app.exception_handler(ModelError)
async def model_error(request, exc):
    return await api_error(request, APIError(422, exc.code, exc.message, exc.detail()))


@app.exception_handler(ValidationError)
async def validation_error(request, exc):
    return await api_error(
        request,
        APIError(
            422,
            "SCHEMA_ERROR",
            "模型结构或必需字段有误",
            json.loads(exc.json(include_input=False)),
        ),
    )


@app.exception_handler(RequestValidationError)
async def request_error(request, exc):
    return await api_error(
        request,
        APIError(
            422,
            "INPUT_ERROR",
            "请求结构不符合要求",
            [dict(loc=e["loc"], msg=e["msg"]) for e in exc.errors()],
        ),
    )


@app.exception_handler(sqlite3.IntegrityError)
async def conflict_error(request, exc):
    return await api_error(request, APIError(409, "CONFLICT", "名称重复或关联数据无效"))


@app.middleware("http")
async def protections(request, call_next):
    request.state.request_id = uid()
    if (
        request.url.path.startswith("/api/")
        and request.headers.get("sec-fetch-site") == "cross-site"
    ):
        return JSONResponse(
            dict(code="ORIGIN_REJECTED", message="跨站请求被拒绝"), status_code=403
        )
    if request.url.path.startswith("/api/") and request.method in (
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
    ):
        origin = request.headers.get("origin")
        allowed = os.getenv(
            "MODELLAB_ORIGINS",
            "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if origin and origin not in allowed:
            return JSONResponse(
                dict(
                    code="ORIGIN_REJECTED",
                    message="跨站请求被拒绝",
                    request_id=request.state.request_id,
                ),
                status_code=403,
            )
        maximum = (
            42 * 1024 * 1024
            if request.url.path.endswith("/assets")
            else 4 * 1024 * 1024
        )
        try:
            if int(request.headers.get("content-length", "0")) > maximum:
                raise ValueError()
        except ValueError:
            return JSONResponse(
                dict(code="BODY_LIMIT", message="请求体过大"), status_code=413
            )
        # Bound chunked bodies as well, before JSON/multipart parsing.
        chunks = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > maximum:
                return JSONResponse(
                    dict(code="BODY_LIMIT", message="请求体过大"), status_code=413
                )
            chunks.append(chunk)
        request._body = b"".join(chunks)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


# Single local workspace; no login, roles, or user-entered credentials.
LOCAL_ACTOR = "local-workspace"


def workspace_session(request: Request):
    token = request.cookies.get("modellab_workspace", "")
    with connect() as db:
        row = db.execute(
            "SELECT csrf FROM sessions WHERE token=? AND user_id=? AND expires>?",
            (hashlib.sha256(token.encode()).hexdigest(), LOCAL_ACTOR, time.time()),
        ).fetchone()
    if not row:
        raise APIError(
            401, "WORKSPACE_SESSION_REQUIRED", "工作台连接已过期，请刷新页面"
        )
    if request.method not in ("GET", "HEAD") and not secrets.compare_digest(
        request.headers.get("X-CSRF-Token", ""), row["csrf"]
    ):
        raise APIError(403, "CSRF_REJECTED", "工作台请求校验失败，请刷新页面")
    return dict(id=LOCAL_ACTOR)


@app.get("/api/workspace")
def workspace(request: Request, response: Response):
    token = request.cookies.get("modellab_workspace", "")
    hashed = hashlib.sha256(token.encode()).hexdigest()
    with transaction() as db:
        row = db.execute(
            "SELECT csrf FROM sessions WHERE token=? AND user_id=? AND expires>?",
            (hashed, LOCAL_ACTOR, time.time()),
        ).fetchone()
        if row:
            return dict(csrf=row["csrf"], mode="personal")
        token = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        # Legacy FK actor only. No password or role is exposed or used by the application.
        db.execute(
            "INSERT OR IGNORE INTO users VALUES (?,?,?,'student',?)",
            (LOCAL_ACTOR, "__modellab_local__", "!disabled", time.time()),
        )
        db.execute("DELETE FROM sessions WHERE expires<?", (time.time(),))
        db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?)",
            (
                hashlib.sha256(token.encode()).hexdigest(),
                LOCAL_ACTOR,
                csrf,
                time.time() + 2592000,
            ),
        )
    response.set_cookie(
        "modellab_workspace",
        token,
        httponly=True,
        samesite="strict",
        secure=os.getenv("COOKIE_SECURE", "0") == "1",
        max_age=2592000,
    )
    return dict(csrf=csrf, mode="personal")


@app.get("/api/settings/model")
def model_settings(user=Depends(workspace_session)):
    return public_connection()


@app.put("/api/settings/model")
def update_model_settings(body: ConnectionInput, user=Depends(workspace_session)):
    try:
        return save_connection(body)
    except ValueError as exc:
        raise APIError(422, "CONNECTION_INVALID", str(exc))


@app.post("/api/settings/model/test")
def test_model_connection(user=Depends(workspace_session)):
    from .providers import test_connection

    try:
        return test_connection()
    except RuntimeError as exc:
        raise APIError(502, "CONNECTION_FAILED", str(exc))


@app.delete("/api/settings/model")
def delete_model_settings(user=Depends(workspace_session)):
    return clear_connection()


@app.get("/api/examples")
def examples(user=Depends(workspace_session)):
    return CATALOG


@app.get("/api/examples/{eid}")
def get_example(eid: str, user=Depends(workspace_session)):
    try:
        return example(eid)
    except KeyError:
        raise APIError(404, "NOT_FOUND", "没有此案例")


class SubmissionInput(Strict):
    title: str = Field(min_length=1, max_length=200)
    example_id: str | None = None


class RenameInput(Strict):
    title: str = Field(min_length=1, max_length=200)


class ImportModelInput(Strict):
    title: str = Field(min_length=1, max_length=200)
    problem_text: str = ""
    model: dict
    data: dict = {}


class RevisionInput(Strict):
    expected_version: int
    problem_text: str = ""
    model: dict
    data: dict = {}
    assets: list[dict] = []


@app.post("/api/submissions")
def create_submission(body: SubmissionInput, user=Depends(workspace_session)):
    id = uid()
    problem = ""
    data = {}
    with transaction() as db:
        item = None
        if body.example_id:
            try:
                item = example(body.example_id)
            except KeyError:
                raise APIError(404, "NOT_FOUND", "没有此案例")
            problem = item["problem_text"]
            data = item["data"]
        db.execute(
            "INSERT INTO submissions VALUES (?,?,?,?,NULL,0,?)",
            (id, user["id"], None, body.title, time.time()),
        )
        s = db.execute("SELECT * FROM submissions WHERE id=?", (id,)).fetchone()
        empty = dict(
            variables=[],
            objective=dict(id="obj", sense="min", expr=dict(kind="number", value=0)),
            constraints=[],
        )
        revision = add_revision(
            db,
            s,
            user,
            RevisionInput(
                expected_version=0,
                problem_text=problem,
                model=item["model"] if item else empty,
                data=data,
            ),
        )
    return dict(id=id, **{"revision": revision})


@app.post("/api/submissions/import")
def import_submission(body: ImportModelInput, user=Depends(workspace_session)):
    id = uid()
    with transaction() as db:
        db.execute(
            "INSERT INTO submissions VALUES (?,?,?,?,NULL,0,?)",
            (id, user["id"], None, body.title.strip(), time.time()),
        )
        s = db.execute("SELECT * FROM submissions WHERE id=?", (id,)).fetchone()
        revision = add_revision(
            db,
            s,
            user,
            RevisionInput(
                expected_version=0,
                problem_text=body.problem_text,
                model=body.model,
                data=body.data,
            ),
            source="import",
        )
    return dict(id=id, revision=revision)


@app.patch("/api/submissions/{sid}")
def rename_submission(sid: str, body: RenameInput, user=Depends(workspace_session)):
    title = body.title.strip()
    if not title:
        raise APIError(422, "INVALID_TITLE", "模型名称不能为空")
    with transaction() as db:
        get_submission(db, sid, user, True)
        db.execute("UPDATE submissions SET title=? WHERE id=?", (title, sid))
        audit(db, sid, user["id"], "submission_renamed", dict(title=title))
    return dict(id=sid, title=title)


@app.delete("/api/submissions/{sid}")
def delete_submission(sid: str, user=Depends(workspace_session)):
    asset_ids = []
    with transaction() as db:
        get_submission(db, sid, user, True)
        rev_rows = db.execute(
            "SELECT id FROM revisions WHERE submission_id=?", (sid,)
        ).fetchall()
        rev_ids = [r["id"] for r in rev_rows]
        if rev_ids:
            marks = ",".join("?" for _ in rev_ids)
            running = db.execute(
                f"SELECT id FROM jobs WHERE revision_id IN ({marks}) AND state IN ('queued','running')",
                rev_ids,
            ).fetchone()
            if running:
                raise APIError(409, "JOB_RUNNING", "有任务正在运行，请先取消后再删除")
            db.execute(f"DELETE FROM jobs WHERE revision_id IN ({marks})", rev_ids)
            db.execute(
                f"DELETE FROM confirmations WHERE revision_id IN ({marks})", rev_ids
            )
        asset_ids = [
            r["id"]
            for r in db.execute(
                "SELECT id FROM assets WHERE submission_id=?", (sid,)
            ).fetchall()
        ]
        db.execute("DELETE FROM ai_threads WHERE submission_id=?", (sid,))
        db.execute("DELETE FROM assets WHERE submission_id=?", (sid,))
        db.execute("DELETE FROM revisions WHERE submission_id=?", (sid,))
        db.execute("DELETE FROM audit WHERE submission_id=?", (sid,))
        db.execute("DELETE FROM submissions WHERE id=?", (sid,))
    for aid in asset_ids:
        path = home() / "assets" / aid
        if path.exists():
            path.unlink()
    return dict(ok=True)


@app.get("/api/submissions")
def submissions(user=Depends(workspace_session)):
    with connect() as db:
        return [
            dict(r)
            for r in db.execute("SELECT * FROM submissions ORDER BY created DESC")
        ]


@app.get("/api/submissions/{sid}")
def submission(sid: str, user=Depends(workspace_session)):
    with connect() as db:
        s = get_submission(db, sid, user)
        revisions = [
            revision_json(r)
            for r in db.execute(
                "SELECT * FROM revisions WHERE submission_id=? ORDER BY number DESC",
                (sid,),
            )
        ]
        for r in revisions:
            c = db.execute(
                "SELECT hash,created FROM confirmations WHERE revision_id=?", (r["id"],)
            ).fetchone()
            r["confirmation"] = dict(c) if c else None
        assets = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM assets WHERE submission_id=? ORDER BY created", (sid,)
            )
        ]
        jobs = [
            public_job(r)
            for r in db.execute(
                "SELECT j.* FROM jobs j JOIN revisions r ON r.id=j.revision_id WHERE r.submission_id=? ORDER BY j.created DESC",
                (sid,),
            )
        ]
        events = [
            {**dict(r), "payload": json.loads(r["payload"])}
            for r in db.execute(
                "SELECT * FROM audit WHERE submission_id=? ORDER BY created DESC",
                (sid,),
            )
        ]
        thread_row = db.execute(
            "SELECT draft_json,messages_json FROM ai_threads WHERE submission_id=?",
            (sid,),
        ).fetchone()
        ai_thread = dict(
            draft=json.loads(thread_row["draft_json"])
            if thread_row and thread_row["draft_json"]
            else None,
            messages=json.loads(thread_row["messages_json"]) if thread_row else [],
        )
    return dict(
        **dict(s),
        revisions=revisions,
        assets=assets,
        jobs=jobs,
        audit=events,
        ai_thread=ai_thread,
        readonly=False,
    )


@app.post("/api/submissions/{sid}/revisions")
def revision(sid: str, body: RevisionInput, user=Depends(workspace_session)):
    with transaction() as db:
        return add_revision(db, get_submission(db, sid, user, True), user, body)


@app.post("/api/revisions/{rid}/validate")
def validate(rid: str, user=Depends(workspace_session)):
    with connect() as db:
        r, s = get_revision(db, rid, user)
    ir = compile_model(json.loads(r["model_json"]), json.loads(r["data_json"]), rid)
    return dict(
        valid=True, variables=len(ir["variables"]), constraints=len(ir["constraints"])
    )


@app.post("/api/revisions/{rid}/confirm")
def confirm(rid: str, user=Depends(workspace_session)):
    with transaction() as db:
        r, s = get_revision(db, rid, user, True, True)
        ir = compile_model(json.loads(r["model_json"]), json.loads(r["data_json"]), rid)
        h = semantic_hash(r)
        db.execute(
            "INSERT OR REPLACE INTO confirmations VALUES (?,?,?,?)",
            (rid, h, user["id"], time.time()),
        )
        audit(db, s["id"], user["id"], "confirmed", dict(revision_id=rid, hash=h))
    return dict(
        hash=h, variables=len(ir["variables"]), constraints=len(ir["constraints"])
    )


@app.post("/api/revisions/{rid}/compile")
def compile_revision(rid: str, user=Depends(workspace_session)):
    with connect() as db:
        r, s = get_revision(db, rid, user, True, True)
        confirmed(db, r)
    ir = compile_model(json.loads(r["model_json"]), json.loads(r["data_json"]), rid)
    code, mapping = generate(ir)
    return dict(
        ir=ir,
        code=code,
        source_map=mapping,
        lp_text=generate_lp(ir),
        stats=model_stats(ir),
    )


@app.get("/api/revisions/{rid}/export-lp")
def export_lp(rid: str, user=Depends(workspace_session)):
    with connect() as db:
        r, s = get_revision(db, rid, user)
        confirmed(db, r)
    ir = compile_model(json.loads(r["model_json"]), json.loads(r["data_json"]), rid)
    lp_text = generate_lp(ir)
    return Response(
        lp_text,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="model-v{r["number"]}.lp"'
        },
    )


class ParseInput(Strict):
    text: str
    model: dict


@app.post("/api/parse")
def parse(body: ParseInput, user=Depends(workspace_session)):
    return dict(expr=parse_expression(body.text, body.model))


@app.post("/api/model/shape")
def model_shape(body: dict, user=Depends(workspace_session)):
    guard_tree(body)
    return ModelDoc.model_validate(body).model_dump()


@app.get("/api/schema")
def schema(user=Depends(workspace_session)):
    return ModelDoc.model_json_schema()


@app.get("/api/fixtures/{name}")
def fixture(name: str, user=Depends(workspace_session)):
    if name not in (
        "production",
        "production_lp",
        "free_variable",
        "assignment",
        "assignment_data",
        "unbounded",
        "infeasible",
        "data_template",
    ):
        raise APIError(404, "NOT_FOUND", "无此样例")
    return json.loads((PROJECT / "fixtures" / f"{name}.json").read_text(encoding="utf-8"))


def capabilities():
    with connect() as db:
        row = db.execute("SELECT * FROM runtime WHERE key=?", ("worker",)).fetchone()
    state = json.loads(row["value"]) if row else {}
    live = bool(row and time.time() - row["updated"] < 15)
    return dict(
        worker=live,
        runner=bool(live and state.get("runner")),
        runner_mode=state.get("runner_mode", "local"),
        runner_reason=state.get("reason", "后台 worker 尚未启动")
        if live
        else "后台 worker 尚未启动或心跳过期",
        solvers=state.get(
            "solvers",
            {"highs": bool(live and state.get("runner")), "gurobi": False, "cplex": False},
        ),
        provider="remote"
        if public_connection()["configured"] and public_connection()["vision_ready"]
        else "unconfigured",
        text_provider="remote"
        if public_connection()["configured"]
        else "unconfigured",
        mock_available=os.getenv("ALLOW_MOCK", "0") == "1",
    )


@app.get("/api/health")
def health():
    return dict(database=True, **capabilities())


class RunInput(Strict):
    request_key: str = Field(min_length=8, max_length=100)
    solver: Literal["highs", "gurobi", "cplex"] = "highs"
    time_limit: float = Field(default=30, gt=0, le=600)
    mip_gap: float | None = Field(default=None, ge=0, le=1)
    threads: int = Field(default=1, ge=0, le=64)
    seed: int = Field(default=0, ge=0, le=2147483647)
    presolve: Literal["auto", "on", "off"] = "auto"


class TranscribeInput(Strict):
    request_key: str = Field(min_length=8, max_length=100)
    provider: str = "remote"


class DiagnoseInput(Strict):
    request_key: str = Field(min_length=8, max_length=100)
    provider: str = "remote"


class TextDraftInput(Strict):
    request_key: str = Field(min_length=8, max_length=100)
    provider: str = "remote"


class AiChatInput(Strict):
    request_key: str = Field(min_length=8, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    provider: str = "remote"


class AnalyzeInput(Strict):
    request_key: str = Field(min_length=8, max_length=100)
    user_request: str = Field(default="", max_length=2000)
    provider: str = "remote"


def public_job(r):
    value = {
        k: r[k]
        for k in (
            "id",
            "revision_id",
            "kind",
            "state",
            "created",
            "started",
            "finished",
            "error",
        )
    }
    value["result"] = json.loads(r["result"]) if r["result"] else None
    return value


def queue(db, r, user, kind, key, payload, h=None):
    old = db.execute(
        "SELECT * FROM jobs WHERE actor=? AND kind=? AND request_key=?",
        (user["id"], kind, key),
    ).fetchone()
    if old:
        if old["revision_id"] != r["id"] or old["payload"] != canonical(payload):
            raise APIError(
                409, "IDEMPOTENCY_CONFLICT", "相同请求标识不能用于不同模型或配置"
            )
        return public_job(old)
    id = uid()
    db.execute(
        "INSERT INTO jobs(id,revision_id,actor,kind,state,request_key,confirmation_hash,payload,created) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            id,
            r["id"],
            user["id"],
            kind,
            "queued",
            key,
            h,
            canonical(payload),
            time.time(),
        ),
    )
    audit(
        db,
        r["submission_id"],
        user["id"],
        "job_queued",
        dict(job_id=id, kind=kind, revision_id=r["id"]),
    )
    return public_job(db.execute("SELECT * FROM jobs WHERE id=?", (id,)).fetchone())


@app.post("/api/revisions/{rid}/runs", status_code=202)
def run(rid: str, body: RunInput, user=Depends(workspace_session)):
    with transaction() as db:
        r, s = get_revision(db, rid, user, True, True)
        c = confirmed(db, r)
        cap = capabilities()
        if not cap["runner"]:
            raise APIError(503, "RUNNER_UNAVAILABLE", cap["runner_reason"])
        if not cap["solvers"].get(body.solver, False):
            pkg = {"highs": "highspy", "gurobi": "gurobipy", "cplex": "cplex"}[body.solver]
            raise APIError(
                422,
                "SOLVER_UNAVAILABLE",
                f"当前环境未检测到 {body.solver.upper()}（需要 Python 包 {pkg}），请先安装或切换为 HiGHS",
            )
        return queue(
            db,
            r,
            user,
            "solve",
            body.request_key,
            dict(
                solver=body.solver,
                time_limit=body.time_limit,
                mip_gap=body.mip_gap,
                threads=body.threads,
                seed=body.seed,
                presolve=body.presolve,
                execution_mode=cap["runner_mode"],
            ),
            c["hash"],
        )


@app.post("/api/revisions/{rid}/transcriptions", status_code=202)
def transcription(rid: str, body: TranscribeInput, user=Depends(workspace_session)):
    with transaction() as db:
        r, s = get_revision(db, rid, user, True, True)
        if not json.loads(r["assets_json"]):
            raise APIError(422, "IMAGES_REQUIRED", "请上传图片并保存当前版本")
        cap = capabilities()
        if not cap["worker"]:
            raise APIError(503, "WORKER_UNAVAILABLE", "后台 worker 未运行")
        if body.provider == "mock" and not cap["mock_available"]:
            raise APIError(422, "MOCK_DISABLED", "需显式设置 ALLOW_MOCK=1")
        if body.provider not in ("remote", "mock"):
            raise APIError(422, "PROVIDER_INVALID", "未知 provider")
        if body.provider == "remote" and cap["provider"] != "remote":
            raise APIError(
                503,
                "PROVIDER_UNCONFIGURED",
                "请在模型设置中配置 API 地址、模型名称、API Key，并启用图像输入",
            )
        return queue(
            db,
            r,
            user,
            "transcribe",
            body.request_key,
            dict(
                provider=body.provider,
                connection_revision=public_connection()["revision"]
                if body.provider == "remote"
                else None,
            ),
        )


@app.post("/api/revisions/{rid}/diagnoses", status_code=202)
def diagnose_revision(
    rid: str, body: DiagnoseInput, user=Depends(workspace_session)
):
    with transaction() as db:
        r, s = get_revision(db, rid, user, True, True)
        cap = capabilities()
        if not cap["worker"]:
            raise APIError(503, "WORKER_UNAVAILABLE", "后台 worker 未运行")
        if body.provider == "mock" and not cap["mock_available"]:
            raise APIError(422, "MOCK_DISABLED", "需显式设置 ALLOW_MOCK=1")
        if body.provider not in ("remote", "mock"):
            raise APIError(422, "PROVIDER_INVALID", "未知 provider")
        if body.provider == "remote" and cap["text_provider"] != "remote":
            raise APIError(
                503,
                "PROVIDER_UNCONFIGURED",
                "请在模型设置中配置 API 地址、模型名称和 API Key",
            )
        return queue(
            db,
            r,
            user,
            "diagnose",
            body.request_key,
            dict(
                provider=body.provider,
                connection_revision=public_connection()["revision"]
                if body.provider == "remote"
                else None,
            ),
        )


@app.post("/api/revisions/{rid}/drafts", status_code=202)
def generate_draft_revision(
    rid: str, body: TextDraftInput, user=Depends(workspace_session)
):
    with transaction() as db:
        r, s = get_revision(db, rid, user, True, True)
        if not (r["problem_text"] or "").strip():
            raise APIError(422, "PROBLEM_TEXT_REQUIRED", "请先填写问题背景并保存当前版本")
        cap = capabilities()
        if not cap["worker"]:
            raise APIError(503, "WORKER_UNAVAILABLE", "后台 worker 未运行")
        if body.provider == "mock" and not cap["mock_available"]:
            raise APIError(422, "MOCK_DISABLED", "需显式设置 ALLOW_MOCK=1")
        if body.provider not in ("remote", "mock"):
            raise APIError(422, "PROVIDER_INVALID", "未知 provider")
        if body.provider == "remote" and cap["text_provider"] != "remote":
            raise APIError(
                503,
                "PROVIDER_UNCONFIGURED",
                "请在模型设置中配置 API 地址、模型名称和 API Key",
            )
        return queue(
            db,
            r,
            user,
            "generate_draft",
            body.request_key,
            dict(
                provider=body.provider,
                connection_revision=public_connection()["revision"]
                if body.provider == "remote"
                else None,
            ),
        )


@app.post("/api/revisions/{rid}/ai-chat", status_code=202)
def chat_draft_revision(
    rid: str, body: AiChatInput, user=Depends(workspace_session)
):
    with transaction() as db:
        r, s = get_revision(db, rid, user, True, True)
        cap = capabilities()
        if not cap["worker"]:
            raise APIError(503, "WORKER_UNAVAILABLE", "后台 worker 未运行")
        if body.provider == "mock" and not cap["mock_available"]:
            raise APIError(422, "MOCK_DISABLED", "需显式设置 ALLOW_MOCK=1")
        if body.provider not in ("remote", "mock"):
            raise APIError(422, "PROVIDER_INVALID", "未知 provider")
        if body.provider == "remote" and cap["text_provider"] != "remote":
            raise APIError(
                503,
                "PROVIDER_UNCONFIGURED",
                "请在模型设置中配置 API 地址、模型名称和 API Key",
            )
        row = db.execute(
            "SELECT draft_json,messages_json FROM ai_threads WHERE submission_id=?",
            (s["id"],),
        ).fetchone()
        msgs = json.loads(row["messages_json"]) if row else []
        msgs.append(
            dict(
                id=uid(),
                role="user",
                content=body.message.strip(),
                created=time.time(),
            )
        )
        draft_json = row["draft_json"] if row else None
        db.execute(
            "INSERT OR REPLACE INTO ai_threads(submission_id,draft_json,messages_json,updated) VALUES (?,?,?,?)",
            (s["id"], draft_json, canonical(msgs), time.time()),
        )
        return queue(
            db,
            r,
            user,
            "chat_draft",
            body.request_key,
            dict(
                message=body.message.strip(),
                provider=body.provider,
                connection_revision=public_connection()["revision"]
                if body.provider == "remote"
                else None,
            ),
        )


@app.delete("/api/submissions/{sid}/ai-thread")
def clear_ai_thread(sid: str, user=Depends(workspace_session)):
    with transaction() as db:
        get_submission(db, sid, user, True)
        db.execute("DELETE FROM ai_threads WHERE submission_id=?", (sid,))
    return dict(ok=True)


@app.post("/api/jobs/{jid}/analyze", status_code=202)
def analyze_solve_job(
    jid: str, body: AnalyzeInput, user=Depends(workspace_session)
):
    with transaction() as db:
        solve_job = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if (
            not solve_job
            or solve_job["kind"] != "solve"
            or solve_job["state"] != "succeeded"
        ):
            raise APIError(409, "RESULT_UNAVAILABLE", "请等待求解完成后再进行 AI 分析")
        r, s = get_revision(db, solve_job["revision_id"], user, True)
        cap = capabilities()
        if not cap["worker"]:
            raise APIError(503, "WORKER_UNAVAILABLE", "后台 worker 未运行")
        if body.provider == "mock" and not cap["mock_available"]:
            raise APIError(422, "MOCK_DISABLED", "需显式设置 ALLOW_MOCK=1")
        if body.provider not in ("remote", "mock"):
            raise APIError(422, "PROVIDER_INVALID", "未知 provider")
        if body.provider == "remote" and cap["text_provider"] != "remote":
            raise APIError(
                503,
                "PROVIDER_UNCONFIGURED",
                "请在模型设置中配置 API 地址、模型名称和 API Key",
            )
        return queue(
            db,
            r,
            user,
            "analyze_result",
            body.request_key,
            dict(
                solve_job_id=jid,
                user_request=body.user_request.strip(),
                provider=body.provider,
                connection_revision=public_connection()["revision"]
                if body.provider == "remote"
                else None,
            ),
        )


@app.get("/api/jobs/{jid}")
def job(jid: str, user=Depends(workspace_session)):
    with connect() as db:
        j = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not j:
            raise APIError(404, "NOT_FOUND", "任务不存在")
        get_revision(db, j["revision_id"], user)
        return public_job(j)


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str, user=Depends(workspace_session)):
    with transaction() as db:
        j = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not j:
            raise APIError(404, "NOT_FOUND", "任务不存在")
        r, s = get_revision(db, j["revision_id"], user, True)
        if j["state"] in ("queued", "running"):
            db.execute(
                "UPDATE jobs SET cancel=1,state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END WHERE id=?",
                (jid,),
            )
            audit(db, s["id"], user["id"], "cancel_requested", dict(job_id=jid))
    return dict(ok=True)


@app.post("/api/jobs/{jid}/adopt")
def adopt(jid: str, body: RevisionInput, user=Depends(workspace_session)):
    with transaction() as db:
        j = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if (
            not j
            or j["kind"] not in ("transcribe", "generate_draft")
            or j["state"] != "succeeded"
        ):
            raise APIError(409, "RESULT_UNAVAILABLE", "模型草稿尚未就绪")
        r, s = get_revision(db, j["revision_id"], user, True, True)
        # User-edited draft is explicit; no asynchronous response may overwrite a newer revision.
        return add_revision(db, s, user, body, source="transcription:" + jid)


@app.post("/api/submissions/{sid}/assets")
async def upload(
    sid: str, files: list[UploadFile] = File(...), user=Depends(workspace_session)
):
    with connect() as db:
        get_submission(db, sid, user, True)
    if not 1 <= len(files) <= 4:
        raise APIError(422, "ASSET_LIMIT", "单次上传 1–4 张图片")
    staged = []
    for f in files:
        content = await f.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024:
            raise APIError(413, "ASSET_LIMIT", "每张图片不超过 10 MiB")
        try:
            with Image.open(io.BytesIO(content)) as img:
                fmt = img.format
                w, h = img.size
                img.verify()
            if fmt not in ("PNG", "JPEG") or w * h > 25_000_000:
                raise ValueError()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            raise APIError(
                422, "IMAGE_INVALID", "仅接受有效 PNG/JPEG，最大 2500 万像素"
            )
        id = uid()
        staged.append(
            (
                id,
                content,
                dict(
                    id=id,
                    submission_id=sid,
                    name=Path(f.filename or "image").name[:200],
                    mime="image/png" if fmt == "PNG" else "image/jpeg",
                    sha256=hashlib.sha256(content).hexdigest(),
                    size=len(content),
                    width=w,
                    height=h,
                    created=time.time(),
                ),
            )
        )
    folder = home() / "assets"
    folder.mkdir(exist_ok=True)
    with transaction() as db:
        get_submission(db, sid, user, True)
        for id, content, meta in staged:
            (folder / id).write_bytes(content)
            db.execute(
                "INSERT INTO assets VALUES (:id,:submission_id,:name,:mime,:sha256,:size,:width,:height,:created)",
                meta,
            )
        audit(db, sid, user["id"], "assets_uploaded", dict(ids=[a[0] for a in staged]))
    return [a[2] for a in staged]


@app.get("/api/assets/{aid}")
def asset(aid: str, user=Depends(workspace_session)):
    with connect() as db:
        a = db.execute("SELECT * FROM assets WHERE id=?", (aid,)).fetchone()
        if not a:
            raise APIError(404, "NOT_FOUND", "文件不存在")
        get_submission(db, a["submission_id"], user)
    return FileResponse(home() / "assets" / a["id"], media_type=a["mime"])


@app.post("/api/submissions/{sid}/data-files")
async def upload_data_files(
    sid: str, files: list[UploadFile] = File(...), user=Depends(workspace_session)
):
    with connect() as db:
        get_submission(db, sid, user, True)
    if not 1 <= len(files) <= 4:
        raise APIError(422, "ASSET_LIMIT", "单次上传 1–4 个数据文件")
    staged = []
    for f in files:
        content = await f.read(5 * 1024 * 1024 + 1)
        if len(content) > 5 * 1024 * 1024:
            raise APIError(413, "FILE_TOO_LARGE", "数据文件不超过 5 MiB")
        name = os.path.basename(f.filename or "data.csv")
        low = name.lower()
        if not low.endswith((".xlsx", ".csv")):
            raise APIError(422, "INVALID_DATA_FILE", "仅支持 .xlsx 或 .csv 文件")
        mime = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if low.endswith(".xlsx")
            else "text/csv"
        )
        try:
            sheets = parse_spreadsheet_bytes(content, name, mime)
        except Exception as exc:
            raise APIError(422, "INVALID_DATA_FILE", f"无法解析数据文件：{exc}")
        id = uid()
        staged.append(
            (
                id,
                content,
                dict(
                    id=id,
                    submission_id=sid,
                    name=name,
                    mime=mime,
                    sha256=hashlib.sha256(content).hexdigest(),
                    size=len(content),
                    width=0,
                    height=0,
                    created=time.time(),
                ),
                len(sheets),
            )
        )
    folder = home() / "assets"
    folder.mkdir(exist_ok=True)
    with transaction() as db:
        get_submission(db, sid, user, True)
        for id, content, meta, _ in staged:
            (folder / id).write_bytes(content)
            db.execute(
                "INSERT INTO assets VALUES (:id,:submission_id,:name,:mime,:sha256,:size,:width,:height,:created)",
                meta,
            )
        audit(
            db,
            sid,
            user["id"],
            "data_files_uploaded",
            dict(ids=[a[0] for a in staged]),
        )
    return [{**a[2], "sheet_count": a[3]} for a in staged]


@app.get("/api/assets/{aid}/preview")
def preview_data_asset(aid: str, user=Depends(workspace_session)):
    with connect() as db:
        a = db.execute("SELECT * FROM assets WHERE id=?", (aid,)).fetchone()
        if not a:
            raise APIError(404, "NOT_FOUND", "文件不存在")
        get_submission(db, a["submission_id"], user)
    path = home() / "assets" / a["id"]
    if not path.exists():
        raise APIError(404, "NOT_FOUND", "文件内容缺失")
    sheets = parse_spreadsheet_bytes(path.read_bytes(), a["name"], a["mime"])
    return dict(id=a["id"], name=a["name"], sheets=sheets)


@app.get("/api/submissions/{sid}/export")
def export(sid: str, revision_id: str | None = None, user=Depends(workspace_session)):
    with connect() as db:
        s = get_submission(db, sid, user)
        r, _ = get_revision(db, revision_id or s["current_revision"], user)
        if r["submission_id"] != sid:
            raise APIError(403, "FORBIDDEN", "版本不属于该作业")
        confirmed(db, r)
        jobs = db.execute(
            "SELECT * FROM jobs WHERE revision_id=? AND kind='solve' AND state='succeeded' ORDER BY finished DESC",
            (r["id"],),
        ).fetchall()
        events = [
            {**dict(e), "payload": json.loads(e["payload"])}
            for e in db.execute("SELECT * FROM audit WHERE submission_id=?", (sid,))
        ]
    ir = compile_model(json.loads(r["model_json"]), json.loads(r["data_json"]), r["id"])
    code, mapping = generate(ir)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in [
            ("symbolic_model.json", json.loads(r["model_json"])),
            ("data.json", json.loads(r["data_json"])),
            ("model_ir.json", ir),
            ("source_map.json", mapping),
            ("audit.json", events),
            ("asset_layout.json", json.loads(r["assets_json"])),
        ]:
            z.writestr(name, canonical(value))
        z.writestr("solve.py", code)
        z.writestr("problem.txt", r["problem_text"])
        req_path = PROJECT / "runner" / "requirements.txt"
        if not req_path.exists():
            req_path = PROJECT / "requirements.txt"
        z.writestr(
            "requirements.txt",
            req_path.read_text(encoding="utf-8")
            if req_path.exists()
            else "highspy>=1.10,<2\n",
        )
        z.writestr(
            "README.txt",
            "使用 Python 3.12，pip install -r requirements.txt；python solve.py。已安装依赖时可离线复现。\n结果只对应此模型，不能证明符合题意。\n",
        )
        if jobs:
            z.writestr("result.json", jobs[0]["result"])
            z.writestr("config.json", jobs[0]["payload"])
        for item in json.loads(r["assets_json"]):
            with connect() as db:
                a = db.execute(
                    "SELECT * FROM assets WHERE id=? AND submission_id=?",
                    (item["id"], sid),
                ).fetchone()
            if a:
                z.write(
                    home() / "assets" / a["id"],
                    "source_images/"
                    + a["id"]
                    + (".png" if a["mime"] == "image/png" else ".jpg"),
                )
    return Response(
        output.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="modellab-{r["number"]}.zip"'
        },
    )


# Dedicated diagnosis is M4 and deliberately absent; no permissive placeholder endpoint.
frontend = PROJECT / "frontend" / "dist"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
