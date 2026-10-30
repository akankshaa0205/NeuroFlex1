from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gdown

ALLOWED_SUFFIXES = {".csv", ".xlsx"}
DEFAULT_MAX_BYTES = 2 * 1024**3


def local_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def download_file(file_id: str, target: Path, retries: int) -> tuple[str, str | None]:
    if target.is_file() and target.stat().st_size > 0:
        return "existing", None
    target.parent.mkdir(parents=True, exist_ok=True)
    error: Exception | None = None
    for attempt in range(retries):
        try:
            result = gdown.download(id=file_id, output=str(target), quiet=True, resume=True)
            if result is not None and target.is_file() and target.stat().st_size > 0:
                return "downloaded", None
            error = RuntimeError("Google Drive returned no downloaded file")
        except Exception as caught:  # noqa: BLE001 - each failed Drive file is audited
            error = caught
        time.sleep(min(30, 2**attempt))
    return "failed", f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume the selected KIMORE file acquisition.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()
    selected = json.loads(args.manifest.read_text(encoding="utf-8"))
    disallowed = [row for row in selected if Path(row["path"][-1]).suffix.lower() not in ALLOWED_SUFFIXES]
    if disallowed:
        raise ValueError(f"Manifest contains {len(disallowed)} non-skeleton/label files")
    results = []
    counts = {"existing": 0, "downloaded": 0, "failed": 0}
    for index, row in enumerate(selected, start=1):
        bytes_on_disk = local_size(args.output)
        if bytes_on_disk >= args.max_bytes:
            raise RuntimeError(
                f"KIMORE download cap reached: {bytes_on_disk} >= {args.max_bytes} bytes"
            )
        target = args.output.joinpath(*row["path"])
        status, error = download_file(row["id"], target, args.retries)
        counts[status] += 1
        results.append({"id": row["id"], "path": row["path"], "status": status, "error": error})
        if index % 50 == 0 or status == "failed" or index == len(selected):
            print(f"complete={index}/{len(selected)} counts={counts} last={status}:{target}", flush=True)
        if index % 100 == 0:
            args.audit.parent.mkdir(parents=True, exist_ok=True)
            args.audit.write_text(json.dumps({"counts": counts, "files": results}, indent=2), encoding="utf-8")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps({"counts": counts, "files": results}, indent=2), encoding="utf-8")
    print(json.dumps(counts))
    return int(counts["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
