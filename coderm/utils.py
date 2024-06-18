from typing import Any, List, Optional, TypeVar
from pathlib import Path
import json
import re
import gzip


T = TypeVar("T")


def chunkify(lst: List[T], n: int) -> List[List[T]]:
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


def markdown_codeblock_extract(response: str) -> str:
    lines = response.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf


def strip_python_comments(code: str) -> str:
    return re.sub(r"#.*", "", code)
