from __future__ import annotations

import os
import platform
import time
import winreg
from ctypes import POINTER, Structure, byref, c_char, c_char_p, c_uint64, c_ulong, c_ushort, c_double
from pathlib import Path

c_bst_ulong = c_uint64
MAX_PATH = 260
MAX_ECU_TASKS = 50


class TTaskInfo(Structure):
    _pack_ = 1
    _fields_ = [
        ("description", c_char_p),
        ("taskId", c_ushort),
        ("taskCycle", c_ulong),
    ]


def _dll_name() -> str:
    return "CANapAPI.dll" if platform.architecture()[0] == "32bit" else "CANapAPI64.dll"


def _registry_canape_base_path() -> str | None:
    try:
        hkey = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\VECTOR\CANape",
            0,
            winreg.KEY_READ | 0x100,
        )
        base_path = winreg.QueryValueEx(hkey, "Path")[0]
        return base_path
    except OSError:
        return None


def get_canape_dll_path() -> Path:
    env_path = os.getenv("CANAPE_DLL_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p

    base = _registry_canape_base_path()
    if base:
        p = Path(base) / "CANapeAPI" / _dll_name()
        if p.exists():
            return p

    raise FileNotFoundError(
        "CANape API DLL not found. Set the CANAPE_DLL_PATH environment variable "
        "or check the CANape installation in the registry."
    )


def _get_dev_name(which_dev, api_lib, canape):
    p_dev_name = c_char_p()
    api_lib.Asap3GetModuleName(canape, which_dev, byref(p_dev_name))
    return p_dev_name.value.decode("utf-8")


def get_module_names(api_lib, canape) -> list[str]:
    temp_dev_count = c_ulong(0)
    api_lib.Asap3GetModuleCount(canape, byref(temp_dev_count))
    names: list[str] = []
    for index in range(0, temp_dev_count.value):
        names.append(_get_dev_name(index, api_lib, canape))
    return names


def get_dev_by_name(dev_name, api_lib, canape, p_error_msg):
    _ = p_error_msg
    names = get_module_names(api_lib, canape)

    for index, current_name in enumerate(names):
        if current_name == dev_name:
            return index

    for index, current_name in enumerate(names):
        if current_name.lower() == dev_name.lower():
            return index

    for index, current_name in enumerate(names):
        if dev_name.lower() in current_name.lower() or current_name.lower() in dev_name.lower():
            return index

    return None


def resolve_device_handle(
    dev_name: str,
    api_lib,
    canape,
    timeout_s: float = 6.0,
    poll_s: float = 0.5,
    auto_first: bool = False,
):
    deadline = time.time() + max(timeout_s, 0.0)
    last_names: list[str] = []

    while True:
        last_names = get_module_names(api_lib, canape)
        handle = get_dev_by_name(dev_name, api_lib, canape, None)
        if handle is not None:
            return handle, last_names[handle], last_names

        if time.time() >= deadline:
            break
        time.sleep(max(poll_s, 0.1))

    if auto_first and last_names:
        return 0, last_names[0], last_names

    return None, None, last_names


def get_next_values(api_lib, canape, device, task_id, timestamp, vallist, n_signals):
    _ = n_signals
    return api_lib.Asap3GetNextSample(canape, device, task_id, timestamp, vallist)
