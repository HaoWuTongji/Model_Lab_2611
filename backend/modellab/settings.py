"""Local model connection. Secrets never appear in public settings or job payloads."""

import json
import os
import tempfile
import threading
from urllib.parse import urlsplit
from typing import Literal
from pydantic import Field, SecretStr, model_validator
from .schema import Strict
from .storage import home, uid

LOCK = threading.Lock()


class ConnectionInput(Strict):
    connection: Literal["compatible", "local"] = "compatible"
    endpoint: str = Field(default="", max_length=2000)
    model: str = Field(default="", max_length=200)
    api_key: SecretStr | None = None
    vision: bool = True
    json_mode: bool = False

    @model_validator(mode="after")
    def validate_connection(self):
        self.endpoint = self.endpoint.strip().rstrip("/")
        self.model = self.model.strip()
        p = urlsplit(self.endpoint)
        if (
            p.scheme not in ("http", "https")
            or not p.hostname
            or p.username
            or p.password
            or p.query
            or p.fragment
        ):
            raise ValueError("请输入不含用户名、密钥或查询参数的 http(s) API 地址")
        if self.connection == "local" and p.hostname not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise ValueError("本地服务须使用 localhost 或回环地址")
        if p.scheme == "http" and p.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("远端服务请使用 HTTPS")
        if not self.model:
            raise ValueError("请填写模型名称")
        if self.endpoint.endswith("/chat/completions"):
            self.endpoint = self.endpoint[: -len("/chat/completions")]
        return self


def load_connection():
    path = home() / "model-service.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(
        connection="compatible",
        endpoint=os.getenv("MODEL_ENDPOINT", ""),
        model=os.getenv("MODEL_NAME", ""),
        api_key=os.getenv("MODEL_API_KEY", ""),
        vision=os.getenv("MODEL_VISION", "0") == "1",
        json_mode=os.getenv("MODEL_JSON_MODE", "0") == "1",
        revision="environment",
    )


def public_connection(config=None):
    c = load_connection() if config is None else config
    return {
        k: c.get(k, v)
        for k, v in dict(
            connection="compatible",
            endpoint="",
            model="",
            vision=True,
            json_mode=False,
            revision="",
        ).items()
    } | dict(
        key_configured=bool(c.get("api_key")),
        configured=bool(
            c.get("endpoint")
            and c.get("model")
            and (c.get("api_key") or c.get("connection") == "local")
        ),
        vision_ready=bool(c.get("vision")),
    )


def write_connection(config):
    folder = home()
    folder.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".model-service-", dir=folder)
    try:
        os.fchmod(fd, 0o600) if hasattr(os, "fchmod") else None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, folder / "model-service.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_connection(body):
    with LOCK:
        old = load_connection()
        key = (
            body.api_key.get_secret_value().strip()
            if body.api_key is not None
            else old.get("api_key", "")
        )
        if (
            body.api_key is None
            and key
            and body.endpoint != old.get("endpoint", "").rstrip("/")
        ):
            raise ValueError("更换 API 地址时请重新输入该服务的 API Key")
        if body.connection == "compatible" and not key:
            raise ValueError("请输入 API Key")
        c = body.model_dump(exclude={"api_key"}) | dict(api_key=key, revision=uid())
        write_connection(c)
        return public_connection(c)


def clear_connection():
    with LOCK:
        os.environ.pop("MODEL_API_KEY", None)
        c = dict(
            connection="compatible",
            endpoint="",
            model="",
            api_key="",
            vision=True,
            json_mode=False,
            revision=uid(),
        )
        write_connection(c)
        return public_connection(c)
