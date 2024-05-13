from typing import Any, List, Optional
from pathlib import Path
import json
import gzip


def chunkify(lst: List[Any], n: int):
    chunks = []
    for i in range(0, len(lst), n):
        chunk = []
        for j in range(n):
            if i + j < len(lst):
                chunk.append(lst[i + j])
        chunks.append(chunk)
    return chunks


def container_restart(name="code-exec", runtime="docker"):
    import subprocess
    p = subprocess.Popen(
        [runtime, "restart", name], stdout=subprocess.PIPE)
    p.communicate()
    return p.returncode


def gunzip_json_read(path: Path) -> Optional[dict]:
    try:
        with gzip.open(path, "rt") as f:
            return json.load(f)
    except Exception as e:
        return None


def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
