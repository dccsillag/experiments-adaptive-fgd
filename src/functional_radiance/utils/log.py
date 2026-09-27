import datetime
import os
import socket
import subprocess


def get_git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_modified_files():
    return (
        subprocess.check_output(["git", "diff", "--name-status"])
        .decode("ascii")
        .strip()
    )


def get_readable_time():
    return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()


def get_hostname():
    return socket.gethostname()


def get_log_info():
    return {
        "git": get_git_revision(),
        "time": get_readable_time(),
        "host": get_hostname(),
        "user": os.getlogin(),
    }
