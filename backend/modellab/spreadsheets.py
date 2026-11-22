"""Safe, bounded Excel (.xlsx) and CSV (.csv) parsing for data attachments."""

from __future__ import annotations
import csv
import io
import math
from pathlib import Path
from .storage import home


def _clean_cell(val):
    if val is None:
        return ""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        if isinstance(val, float) and not math.isfinite(val):
            return ""
        return int(val) if isinstance(val, float) and val.is_integer() else val
    return str(val).strip()


def _trim_empty_rows(rows: list[list]) -> list[list]:
    while rows and all(c == "" for c in rows[-1]):
        rows.pop()
    return rows


def parse_spreadsheet_bytes(
    raw: bytes,
    filename: str,
    mime: str,
    max_rows: int = 120,
    max_cols: int = 30,
) -> list[dict]:
    low = (filename or "").lower()
    if mime == "text/csv" or low.endswith(".csv"):
        text = None
        for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise ValueError("CSV 文件编码无法识别")
        reader = csv.reader(io.StringIO(text))
        rows = []
        total = 0
        for r in reader:
            total += 1
            if len(rows) < max_rows:
                rows.append([_clean_cell(c) for c in r[:max_cols]])
        rows = _trim_empty_rows(rows)
        return [
            dict(
                sheet_name=Path(filename).stem or "CSV",
                rows=rows,
                total_rows=total,
                truncated=total > max_rows,
            )
        ]

    if (
        mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        or low.endswith(".xlsx")
    ):
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        try:
            sheets = []
            for sname in wb.sheetnames[:10]:
                ws = wb[sname]
                rows = []
                total = 0
                for r in ws.iter_rows(values_only=True):
                    total += 1
                    if len(rows) < max_rows:
                        rows.append([_clean_cell(c) for c in r[:max_cols]])
                rows = _trim_empty_rows(rows)
                if rows:
                    sheets.append(
                        dict(
                            sheet_name=sname,
                            rows=rows,
                            total_rows=total,
                            truncated=total > max_rows,
                        )
                    )
            return sheets
        finally:
            wb.close()

    raise ValueError("仅支持 .xlsx 或 .csv 格式的数据文件")


def load_asset_tables(assets_list: list[dict], db) -> list[dict]:
    """Load and parse all spreadsheet attachments referenced by a revision."""
    tables = []
    for item in assets_list or []:
        aid = item.get("id")
        if not aid:
            continue
        row = db.execute(
            "SELECT id,name,mime FROM assets WHERE id=?", (aid,)
        ).fetchone()
        if not row:
            continue
        mime = row["mime"]
        name = row["name"]
        if mime not in (
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ) and not name.lower().endswith((".csv", ".xlsx")):
            continue
        path = home() / "assets" / aid
        if not path.exists():
            continue
        sheets = parse_spreadsheet_bytes(path.read_bytes(), name, mime)
        tables.append(dict(asset_id=aid, filename=name, sheets=sheets))
    return tables
