def chunkify(lst, n):
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
