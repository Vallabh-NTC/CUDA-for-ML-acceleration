from __future__ import annotations

import argparse
import json
import logging
import platform
import socket
import sys
import time
from ctypes import POINTER, byref, c_bool, c_double, c_ulong, c_ushort, windll
from pathlib import Path

from canape_api_local import (
    MAX_ECU_TASKS,
    TTaskInfo,
    c_bst_ulong,
    get_module_names,
    get_canape_dll_path,
    get_next_values,
    resolve_device_handle,
)


def _default_canape_folder() -> Path:
    return Path(__file__).resolve().parent / "config" / "e3" / "canape"


def _read_signal_names_from_file(signals_file: Path) -> list[str]:
    if not signals_file.exists():
        raise FileNotFoundError(f"Signal list file not found: {signals_file}")

    signal_names: list[str] = []
    for raw_line in signals_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        signal_names.extend(token for token in line.split() if token.strip())

    if not signal_names:
        raise ValueError(f"Signal list file is empty or contains no valid signals: {signals_file}")

    return signal_names


def _normalize_signal_name(signal_name: str) -> str:
    raw = signal_name.strip()
    if "::" in raw:
        return raw.split("::")[-1]
    return raw


class CanapeE3SignalAcquisition:
    def __init__(
        self,
        signal_names: list[str],
        canape_device_name: str,
        canape_folder: Path,
        cna_file: Path | None,
        sample_interval: float,
        diagnostics: bool,
        status_interval: float,
        asap_modal_mode: bool,
        device_discovery_timeout: float,
        auto_device: bool,
        list_modules_only: bool,
        udp_host: str = "",
        udp_port: int = 5005,
    ):
        self.signal_names = [_normalize_signal_name(name) for name in signal_names]
        self.canape_device_name = canape_device_name
        self.canape_folder = canape_folder
        self.cna_file = cna_file
        self.sample_interval = sample_interval
        self.diagnostics = diagnostics
        self.status_interval = status_interval
        self.asap_modal_mode = asap_modal_mode
        self.device_discovery_timeout = device_discovery_timeout
        self.auto_device = auto_device
        self.list_modules_only = list_modules_only
        self.modules_only_completed = False
        self.udp_host = udp_host
        self.udp_port = udp_port

        self.api_lib = None
        self.canape = None
        self.com_device = None
        self.com_task = None
        self._udp_sock: socket.socket | None = None

    def initialize(self) -> None:
        if platform.system().lower() != "windows":
            raise RuntimeError("This script works only on Windows (CANape API via DLL).")

        if not self.canape_folder.exists():
            raise FileNotFoundError(f"CANape folder not found: {self.canape_folder}")

        if self.cna_file and not self.cna_file.exists():
            raise FileNotFoundError(f"CNA file not found: {self.cna_file}")

        print("[1/5] Searching for CANape DLL...")
        dll_path = get_canape_dll_path()
        print(f"      DLL found: {dll_path}")

        print("[2/5] Loading DLL...")
        self.api_lib = windll.LoadLibrary(str(dll_path))

        self.canape = c_bst_ulong(0) if platform.architecture()[0] != "32bit" else c_ulong(0)
        g_api_timeout = c_ulong(500000)
        g_fifo_size = c_ulong(9999)
        g_sample_size = c_ulong(128)
        g_debug_mode = c_bool(True)
        g_clear_dev_list = c_bool(False)
        g_hex_mode = c_bool(False)
        g_modal_mode = c_bool(self.asap_modal_mode)

        print("[3/5] Initializing CANape session (Asap3Init5)...")
        init_ret = self.api_lib.Asap3Init5(
            byref(self.canape),
            g_api_timeout,
            str(self.canape_folder).encode("utf-8"),
            g_fifo_size,
            g_sample_size,
            g_debug_mode,
            g_clear_dev_list,
            g_hex_mode,
            g_modal_mode,
        )
        if init_ret == 0:
            raise RuntimeError(
                "Asap3Init5 returned error (ret=0). "
                "Check CANape/CNA project and try with --asap-modal if necessary."
            )
        self.api_lib.Asap3UseNAN(self.canape, c_bool(False))
        print(f"      CANape session initialized (ret={init_ret}, modal={self.asap_modal_mode})")

        if self.cna_file:
            print(f"      Loading CNA: {self.cna_file}")
            load_ret = self.api_lib.Asap3LoadCNAFile(self.canape, str(self.cna_file).encode("utf-8"))
            print(f"      CNA loaded (ret={load_ret})")
        else:
            print("      CNA not specified")

        if self.list_modules_only:
            modules = get_module_names(self.api_lib, self.canape)
            print("[modules] Modules exposed by CANape:")
            if modules:
                for index, module_name in enumerate(modules):
                    print(f"  - [{index}] {module_name}")
            else:
                print("  - (no modules)")
            self.modules_only_completed = True
            return

        print("[4/5] Configuring device/task...")
        self._configure_device_and_task()
        print("[5/5] Setting up acquisition channel...")
        self._setup_signal_channel()
        print("      Setup completed.")

        if self.udp_host:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"      UDP output active -> {self.udp_host}:{self.udp_port}")

        if self.diagnostics:
            self._print_diagnostics()

    def _configure_device_and_task(self) -> None:
        self.com_device, resolved_name, module_names = resolve_device_handle(
            dev_name=self.canape_device_name,
            api_lib=self.api_lib,
            canape=self.canape,
            timeout_s=self.device_discovery_timeout,
            poll_s=0.5,
            auto_first=self.auto_device,
        )

        if self.com_device is None and len(module_names) == 1:
            self.com_device = 0
            resolved_name = module_names[0]
            print(
                f"      Device auto-selected (single module): '{resolved_name}' "
                f"(requested: '{self.canape_device_name}')"
            )

        if self.com_device is None:
            known_modules = ", ".join(module_names) if module_names else "<none>"
            raise RuntimeError(
                f"CANape device '{self.canape_device_name}' not found. "
                f"Available modules: {known_modules}. "
                "Check the device name in the project/CNA or use --auto-device."
            )
        if resolved_name and resolved_name != self.canape_device_name:
            print(f"      Device resolved via fallback: '{self.canape_device_name}' -> '{resolved_name}'")

        my_no_tasks = c_ushort(0)
        my_max_tasks_info = c_ushort(MAX_ECU_TASKS)

        self.api_lib.Asap3GetEcuTasks(self.canape, self.com_device, None, byref(my_no_tasks), my_max_tasks_info)
        my_obj_t_task_info = (TTaskInfo * my_no_tasks.value)()
        self.api_lib.Asap3GetEcuTasks(
            self.canape,
            self.com_device,
            byref(my_obj_t_task_info),
            byref(my_no_tasks),
            my_max_tasks_info,
        )

        if len(my_obj_t_task_info) < 1:
            raise RuntimeError("CANape does not expose any available ECU tasks.")

        self.com_task = my_obj_t_task_info[0]
        print(
            f"      Device={self.canape_device_name} handle={self.com_device} | "
            f"TaskId={self.com_task.taskId} Cycle={self.com_task.taskCycle}"
        )

    def _setup_signal_channel(self) -> None:
        for signal_name in self.signal_names:
            ret = self.api_lib.Asap3SetupDataAcquisitionChnl(
                self.canape,
                self.com_device,
                signal_name.encode("utf-8"),
                c_ushort(1),
                c_ushort(self.com_task.taskId),
                c_ushort(0),
                True,
            )
            print(f"      Canale '{signal_name}' configurato (ret={ret})")

    def _print_diagnostics(self) -> None:
        dev_count = c_ulong(0)
        self.api_lib.Asap3GetModuleCount(self.canape, byref(dev_count))
        print(f"[diag] Number of CANape modules: {dev_count.value}")
        print(f"[diag] Requested signals (normalized): {', '.join(self.signal_names)}")

    def run_loop(self) -> None:
        print("[run] Starting CANape E3 acquisition...")
        self.api_lib.Asap3StartDataAcquisition(self.canape)
        time.sleep(2)

        print(f"Acquisition active. Signals: {', '.join(self.signal_names)}")
        print("Press CTRL+C to terminate.\n")

        timestamp_ms = c_ulong(0)
        last_status = 0.0

        while True:
            loop_start = time.time()

            is_overrun = self.api_lib.Asap3CheckOverrun(self.canape, self.com_device, self.com_task.taskId, True)
            if not is_overrun:
                nr_samples = self.api_lib.Asap3GetFifoLevel(self.canape, self.com_device, c_ushort(self.com_task.taskId))
                if nr_samples <= 0:
                    now = time.time()
                    if (now - last_status) >= self.status_interval:
                        zero_pairs = " | ".join(f"{name}=0" for name in self.signal_names)
                        print(f"[status] No samples available (FIFO={nr_samples}) on task {self.com_task.taskId}")
                        print(f"0.000s | {zero_pairs}")
                        if self._udp_sock is not None:
                            payload: dict = {"t": 0.0}
                            payload.update({name: 0 for name in self.signal_names})
                            self._udp_sock.sendto(
                                json.dumps(payload).encode("utf-8"),
                                (self.udp_host, self.udp_port),
                            )
                        last_status = now

                while nr_samples > 0:
                    nr_signals = len(self.signal_names)
                    sample_fifo = POINTER(c_double * nr_signals)()
                    get_next_values(
                        self.api_lib,
                        self.canape,
                        self.com_device,
                        c_ushort(self.com_task.taskId),
                        byref(timestamp_ms),
                        byref(sample_fifo),
                        nr_signals,
                    )
                    timestamp_s = timestamp_ms.value / 1e4
                    values = sample_fifo.contents
                    value_pairs = [f"{name}={values[index]}" for index, name in enumerate(self.signal_names)]
                    # print signals
                    print(f"{timestamp_s:.3f}s | " + " | ".join(value_pairs))
                    if self._udp_sock is not None:
                        payload = {"t": round(timestamp_s, 4)}
                        payload.update({name: values[index] for index, name in enumerate(self.signal_names)})
                        self._udp_sock.sendto(
                            json.dumps(payload).encode("utf-8"),
                            (self.udp_host, self.udp_port),
                        )
                    nr_samples -= 1
            else:
                now = time.time()
                if (now - last_status) >= self.status_interval:
                    print("[status] Overrun FIFO detected")
                    last_status = now

            elapsed = time.time() - loop_start
            to_wait = self.sample_interval - elapsed
            if to_wait > 0:
                time.sleep(to_wait)

    def shutdown(self) -> None:
        if not self.api_lib or self.canape is None:
            return

        try:
            if hasattr(self.api_lib, "Asap3StopDataAcquisition"):
                self.api_lib.Asap3StopDataAcquisition(self.canape)
        except BaseException:
            pass

        try:
            if hasattr(self.api_lib, "Asap3Exit"):
                self.api_lib.Asap3Exit(self.canape)
        except BaseException:
            pass

        if self._udp_sock is not None:
            self._udp_sock.close()
            self._udp_sock = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously reads one or more signals from CANape (E3) and prints them to the console.",
    )
    parser.add_argument(
        "--signal",
        default=[],
        nargs="+",
        help="One or more signal names (e.g., VehV_v VehAX_ax)",
    )
    parser.add_argument(
        "--signals-file",
        default=str(Path(__file__).resolve().parent / "signals.txt"),
        help="Path to a text file with a list of signals (one per line or separated by space; # for comments)",
    )
    parser.add_argument(
        "--canape-device",
        default="e3",
        help="CANape device name (default: e3)",
    )
    parser.add_argument(
        "--canape-folder",
        default=str(_default_canape_folder()),
        help="CANape project folder used by Asap3Init5",
    )
    parser.add_argument(
        "--cna-file",
        default="",
        help="Optional CNA file path",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Loop interval in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--diagnostics",
        "--diagnostic",
        action="store_true",
        help="Print extra diagnostics on modules/device/task",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="Seconds between status messages when no samples are received (default: 2)",
    )
    parser.add_argument(
        "--asap-modal",
        dest="asap_modal",
        action="store_true",
        default=True,
        help="Use modal mode in Asap3Init5 (default: ON)",
    )
    parser.add_argument(
        "--no-asap-modal",
        dest="asap_modal",
        action="store_false",
        help="Disable modal mode in Asap3Init5 (fallback)",
    )
    parser.add_argument(
        "--device-discovery-timeout",
        type=float,
        default=6.0,
        help="Maximum seconds to find the device after init (default: 6)",
    )
    parser.add_argument(
        "--auto-device",
        action="store_true",
        help="If the requested device is not found, automatically use the first available module",
    )
    parser.add_argument(
        "--list-modules-only",
        action="store_true",
        help="Initialize CANape, print the exposed modules, and exit",
    )
    parser.add_argument(
        "--udp-host",
        default="",
        help="UDP destination IP for JSON (e.g., 192.168.1.20). If omitted, UDP sending is disabled.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5005,
        help="UDP destination port (default: 5005)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    signal_names_input = list(args.signal)
    if args.signals_file:
        signals_path = Path(args.signals_file).resolve()
        if signals_path.exists():
            signal_names_input.extend(_read_signal_names_from_file(signals_path))
        elif not signal_names_input:
            raise SystemExit(f"Signals file not found: {signals_path}")

    signal_names = []
    seen: set[str] = set()
    for signal_name in signal_names_input:
        normalized = _normalize_signal_name(signal_name)
        if normalized not in seen:
            seen.add(normalized)
            signal_names.append(normalized)

    canape_folder = Path(args.canape_folder).resolve()
    cna_file = Path(args.cna_file).resolve() if args.cna_file else None

    try:
        if not signal_names and not args.list_modules_only:
            raise ValueError("No signals specified: use --signal and/or --signals-file")

        acq = CanapeE3SignalAcquisition(
            signal_names=signal_names,
            canape_device_name=args.canape_device,
            canape_folder=canape_folder,
            cna_file=cna_file,
            sample_interval=args.sample_interval,
            diagnostics=args.diagnostics,
            status_interval=args.status_interval,
            asap_modal_mode=args.asap_modal,
            device_discovery_timeout=args.device_discovery_timeout,
            auto_device=args.auto_device,
            list_modules_only=args.list_modules_only,
            udp_host=args.udp_host,
            udp_port=args.udp_port,
        )

        acq.initialize()
        if acq.modules_only_completed:
            return 0

        print(f"Signals to acquire ({len(signal_names)}): {', '.join(signal_names)}")

        try:
            acq.run_loop()
        except KeyboardInterrupt:
            print("\nUser requested interruption.")
        finally:
            acq.shutdown()

    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
