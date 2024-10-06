import platform

def is_wsl(v: str = platform.uname().release) -> int:
    # WSL v1 and v2
    if v.endswith("-Microsoft") or v.endswith("microsoft-standard-WSL2"):
        return True

    return False
