import shutil
import mmap
import errno
import os

def cleanup(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    mkdir_p(dirname)




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def file_get_contents(filename):
    with open(filename, encoding="utf-8", errors='ignore') as f:
        return f.read()


def mapcount(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines
