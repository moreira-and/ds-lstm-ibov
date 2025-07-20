from git import subprocess

def getgit_commit_hash(self):
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return commit_hash
    except Exception:
        return "unavailable"