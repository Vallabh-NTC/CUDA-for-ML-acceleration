from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from ctypes import POINTER, byref, c_bool, c_char_p, c_double, c_ulong, c_ushort, windll
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
    return Path(__file__).resolve().parent / "config" / "e3" / "canape" / "vn5650_smotion"


def _read_signal_names_from_file(signals_file: Path) -> list[str]:
    if not signals_file.exists():
        raise FileNotFoundError(f"File lista segnali non trovato: {signals_file}")

    signal_names: list[str] = []
    for raw_line in signals_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#")[0].strip()
        if not line:
            continue
        signal_names.extend(token for token in line.split() if token.strip())

    if not signal_names:
        raise ValueError(f"File lista segnali vuoto o senza segnali validi: {signals_file}")

    return signal_names


def _normalize_signal_name(signal_name: str) -> str:
    raw = signal_name.strip()
    if "::" in raw:
        return raw.split("::")[-1]
    return raw


def _deduplicate_signal_names(signal_names_input: list[str]) -> list[str]:
    signal_names: list[str] = []
    seen: set[str] = set()
    for signal_name in signal_names_input:
        normalized = _normalize_signal_name(signal_name)
        if normalized not in seen:
            seen.add(normalized)
            signal_names.append(normalized)
    return signal_names


def _get_canape_process_pids() -> set[int]:
    pids: set[int] = set()
    for image_name in ("CANape64.exe", "CANape.exe"):
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) < 2:
                continue
            if row[0].strip().lower() != image_name.lower():
                continue
            try:
                pids.add(int(row[1].replace(",", "").strip()))
            except ValueError:
                continue
    return pids


class CanapeE3SignalAcquisition:
    def __init__(
        self,
        device_signal_configs: list[tuple[str, list[str]]],
        canape_folder: Path,
        cna_file: Path | None,
        sample_interval: float,
        diagnostics: bool,
        status_interval: float,
        asap_modal_mode: bool,
        device_discovery_timeout: float,
        auto_device: bool,
        list_modules_only: bool,
        udp_host: str,
        udp_port: int,
    ):
        self.device_signal_configs: list[dict[str, object]] = []
        for device_name, signal_names in device_signal_configs:
            normalized = _deduplicate_signal_names(signal_names)
            if not normalized:
                continue
            self.device_signal_configs.append(
                {
                    "device_name": device_name,
                    "signal_names": normalized,
                    "configured_signal_names": [],
                    "resolved_name": device_name,
                    "com_device": None,
                    "task_id": None,
                    "task_cycle": None,
                }
            )

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
        self._dll_dir_handle = None
        self._canape_pids_before_init: set[int] = set()
        self._canape_pids_started_by_this_process: set[int] = set()
        self._udp_sock: socket.socket | None = None

    def _get_asap3_error_details(self) -> str:
        if not self.api_lib or self.canape is None:
            return ""
        if not hasattr(self.api_lib, "Asap3GetLastError"):
            return ""

        try:
            err_code = int(self.api_lib.Asap3GetLastError(self.canape))
        except BaseException:
            return ""

        err_text = ""
        if hasattr(self.api_lib, "Asap3ErrorText"):
            try:
                err_msg = c_char_p()
                ok = self.api_lib.Asap3ErrorText(self.canape, c_ushort(err_code), byref(err_msg))
                if ok and err_msg.value:
                    err_text = err_msg.value.decode("utf-8", errors="ignore")
            except BaseException:
                pass

        if err_text:
            return f"{err_code} ({err_text})"
        return str(err_code)

    def initialize(self) -> None:
        if platform.system().lower() != "windows":
            raise RuntimeError("Questo script funziona solo su Windows (API CANape via DLL).")

        if not self.canape_folder.exists():
            raise FileNotFoundError(f"Cartella CANape non trovata: {self.canape_folder}")

        if self.cna_file and not self.cna_file.exists():
            raise FileNotFoundError(f"File CNA non trovato: {self.cna_file}")

        print("[1/5] Ricerca DLL CANape...")
        dll_path = get_canape_dll_path()
        print(f"      DLL trovata: {dll_path}")

        print("[2/5] Caricamento DLL...")
        if hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(dll_path.parent))
        self.api_lib = windll.LoadLibrary(str(dll_path))

        self.canape = c_bst_ulong(0) if platform.architecture()[0] != "32bit" else c_ulong(0)
        g_api_timeout = c_ulong(500000)
        g_fifo_size = c_ulong(9999)
        g_sample_size = c_ulong(128)
        g_debug_mode = c_bool(True)
        g_clear_dev_list = c_bool(False)
        g_hex_mode = c_bool(False)
        g_modal_mode = c_bool(self.asap_modal_mode)

        print("[3/5] Inizializzazione sessione CANape (Asap3Init5)...")
        self._canape_pids_before_init = _get_canape_process_pids()
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
            error_details = self._get_asap3_error_details()
            details_suffix = f" Dettaglio ASAP3: {error_details}" if error_details else ""
            raise RuntimeError(
                "Asap3Init5 ha restituito errore (ret=0). "
                "Verifica progetto CANape/CNA e prova con --asap-modal se necessario."
                f"{details_suffix}"
            )

        pids_after_init = _get_canape_process_pids()
        self._canape_pids_started_by_this_process = pids_after_init - self._canape_pids_before_init

        self.api_lib.Asap3UseNAN(self.canape, c_bool(False))
        print(f"      Sessione CANape inizializzata (ret={init_ret}, modal={self.asap_modal_mode})")

        if self.cna_file:
            print(f"      Caricamento CNA: {self.cna_file}")
            load_ret = self.api_lib.Asap3LoadCNAFile(self.canape, str(self.cna_file).encode("utf-8"))
            print(f"      CNA caricato (ret={load_ret})")
        else:
            print("      CNA non specificato")

        if self.list_modules_only:
            modules = get_module_names(self.api_lib, self.canape)
            print("[modules] Moduli esposti da CANape:")
            if modules:
                for index, module_name in enumerate(modules):
                    print(f"  - [{index}] {module_name}")
            else:
                print("  - (nessun modulo)")
            self.modules_only_completed = True
            return

        print("[4/5] Configurazione device/task...")
        self._configure_devices_and_tasks()
        print("[5/5] Setup canali acquisizione...")
        self._setup_signal_channels()
        print("      Setup completato.")

        if self.udp_host:
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"      Invio UDP attivo -> {self.udp_host}:{self.udp_port}")

        if self.diagnostics:
            self._print_diagnostics()

    def _configure_single_device_and_task(self, cfg: dict[str, object]) -> None:
        device_name = str(cfg["device_name"])
        com_device, resolved_name, module_names = resolve_device_handle(
            dev_name=device_name,
            api_lib=self.api_lib,
            canape=self.canape,
            timeout_s=self.device_discovery_timeout,
            poll_s=0.5,
            auto_first=self.auto_device,
        )

        if com_device is None and len(module_names) == 1:
            com_device = 0
            resolved_name = module_names[0]
            print(
                f"      Device auto-selezionato (modulo unico): '{resolved_name}' "
                f"(richiesto: '{device_name}')"
            )

        if com_device is None:
            known_modules = ", ".join(module_names) if module_names else "<none>"
            raise RuntimeError(
                f"Device CANape '{device_name}' non trovato. "
                f"Moduli disponibili: {known_modules}. "
                "Controlla il nome device in progetto/CNA o usa --auto-device."
            )
        if resolved_name and resolved_name != device_name:
            print(f"      Device risolto via fallback: '{device_name}' -> '{resolved_name}'")

        my_no_tasks = c_ushort(0)
        my_max_tasks_info = c_ushort(MAX_ECU_TASKS)

        self.api_lib.Asap3GetEcuTasks(self.canape, com_device, None, byref(my_no_tasks), my_max_tasks_info)
        my_obj_t_task_info = (TTaskInfo * my_no_tasks.value)()
        self.api_lib.Asap3GetEcuTasks(
            self.canape,
            com_device,
            byref(my_obj_t_task_info),
            byref(my_no_tasks),
            my_max_tasks_info,
        )

        if len(my_obj_t_task_info) < 1:
            raise RuntimeError("CANape non espone ECU task disponibili.")

        com_task = my_obj_t_task_info[0]
        cfg["resolved_name"] = resolved_name or device_name
        cfg["com_device"] = com_device
        cfg["task_id"] = int(com_task.taskId)
        cfg["task_cycle"] = int(com_task.taskCycle)
        print(
            f"      Device={device_name} handle={com_device} | "
            f"TaskId={com_task.taskId} Cycle={com_task.taskCycle}"
        )

    def _configure_devices_and_tasks(self) -> None:
        if not self.device_signal_configs:
            raise RuntimeError("Nessuna configurazione device/segnali disponibile.")

        for cfg in self.device_signal_configs:
            self._configure_single_device_and_task(cfg)

    def _setup_signal_channels(self) -> None:
        for cfg in self.device_signal_configs:
            device_name = str(cfg["device_name"])
            com_device = cfg["com_device"]
            task_id = int(cfg["task_id"])
            signal_names = list(cfg["signal_names"])
            configured_signal_names: list[str] = []

            for signal_name in signal_names:
                tried_names = [signal_name]
                qualified_name = f"{device_name}::{signal_name}"
                if "::" not in signal_name:
                    tried_names.append(qualified_name)

                ret = 0
                used_name = signal_name
                for candidate_name in tried_names:
                    ret = self.api_lib.Asap3SetupDataAcquisitionChnl(
                        self.canape,
                        com_device,
                        candidate_name.encode("utf-8"),
                        c_ushort(1),
                        c_ushort(task_id),
                        c_ushort(0),
                        True,
                    )
                    if ret:
                        used_name = candidate_name
                        break

                if not ret:
                    tried = ", ".join(tried_names)
                    print(
                        f"[warn] Canale '{signal_name}' non configurato su '{device_name}'. "
                        f"Nomi provati: {tried}. Valore mantenuto a 0."
                    )
                    continue

                configured_signal_names.append(signal_name)

                print(
                    f"      Canale '{signal_name}' configurato su '{device_name}' "
                    f"(nome usato: '{used_name}', ret={ret})"
                )

            cfg["configured_signal_names"] = configured_signal_names

    def _print_diagnostics(self) -> None:
        dev_count = c_ulong(0)
        self.api_lib.Asap3GetModuleCount(self.canape, byref(dev_count))
        print(f"[diag] Numero moduli CANape: {dev_count.value}")
        for cfg in self.device_signal_configs:
            device_name = str(cfg["device_name"])
            signal_names = list(cfg["signal_names"])
            print(f"[diag] Device {device_name}: {', '.join(signal_names)}")

    def run_loop(self) -> None:
        print("[run] Avvio acquisizione CANape E3...")
        self.api_lib.Asap3StartDataAcquisition(self.canape)
        time.sleep(2)
        run_start = time.time()

        print("Acquisizione attiva. Configurazioni per device:")
        for cfg in self.device_signal_configs:
            device_name = str(cfg["device_name"])
            signal_names = list(cfg["signal_names"])
            print(f"  - {device_name}: {', '.join(signal_names)}")
        print("Premi CTRL+C per terminare.\n")

        timestamp_ms = c_ulong(0)
        last_status_by_device = {str(cfg["device_name"]): 0.0 for cfg in self.device_signal_configs}
        last_print_wallclock = run_start
        last_sample_wallclock_by_device = {
            str(cfg["device_name"]): run_start for cfg in self.device_signal_configs
        }
        device_marked_disconnected = {
            str(cfg["device_name"]): False for cfg in self.device_signal_configs
        }
        disconnect_timeout_s = max(5.0, self.status_interval * 3.0)

        # Build signal order and prefix map for UDP payload
        signal_order = []
        signal_prefix_map = {}
        for cfg in self.device_signal_configs:
            device_name = str(cfg["device_name"])
            prefix = "eth_" if device_name.lower().startswith("e3") else ("smotion_" if device_name.lower().startswith("smotion") else f"{device_name.lower()}_")
            for signal_name in list(cfg["signal_names"]):
                signal_order.append(signal_name)
                signal_prefix_map[signal_name] = prefix
        latest_values: dict[str, float] = {signal_name: 0.0 for signal_name in signal_order}
        latest_timestamp_s = 0.0

        while True:
            loop_start = time.time()

            for cfg in self.device_signal_configs:
                device_name = str(cfg["device_name"])
                resolved_name = str(cfg["resolved_name"])
                com_device = cfg["com_device"]
                task_id = int(cfg["task_id"])
                active_signal_names = list(cfg["configured_signal_names"])

                if not active_signal_names:
                    now = time.time()
                    if (now - last_status_by_device[device_name]) >= self.status_interval:
                        print(
                            f"[status] [{resolved_name}] Nessun canale attivo configurato. "
                            "I segnali richiesti restano a 0."
                        )
                        last_status_by_device[device_name] = now
                    continue

                is_overrun = self.api_lib.Asap3CheckOverrun(self.canape, com_device, task_id, True)
                if is_overrun:
                    now = time.time()
                    if (now - last_status_by_device[device_name]) >= self.status_interval:
                        print(f"[status] [{resolved_name}] Overrun FIFO rilevato")
                        last_status_by_device[device_name] = now

                nr_samples = self.api_lib.Asap3GetFifoLevel(self.canape, com_device, c_ushort(task_id))
                if nr_samples <= 0:
                    now = time.time()
                    if (now - last_sample_wallclock_by_device[device_name]) >= disconnect_timeout_s:
                        for signal_name in list(cfg["signal_names"]):
                            latest_values[signal_name] = 0.0
                        if not device_marked_disconnected[device_name]:
                            print(
                                f"[status] [{resolved_name}] Nessun campione da {disconnect_timeout_s:.1f}s: "
                                "valori azzerati."
                            )
                            device_marked_disconnected[device_name] = True

                    if (now - last_status_by_device[device_name]) >= self.status_interval:
                        print(
                            f"[status] [{resolved_name}] Nessun campione disponibile "
                            f"(FIFO={nr_samples}) sul task {task_id}"
                        )
                        last_status_by_device[device_name] = now
                    continue

                device_latest_values: list[float] | None = None
                device_latest_timestamp_s = 0.0
                remaining = nr_samples
                while remaining > 0:
                    nr_signals = len(active_signal_names)
                    sample_fifo = POINTER(c_double * nr_signals)()
                    get_next_values(
                        self.api_lib,
                        self.canape,
                        com_device,
                        c_ushort(task_id),
                        byref(timestamp_ms),
                        byref(sample_fifo),
                        nr_signals,
                    )
                    device_latest_timestamp_s = timestamp_ms.value / 1e4
                    values = sample_fifo.contents
                    device_latest_values = [values[index] for index in range(nr_signals)]
                    remaining -= 1

                if device_latest_values is not None:
                    for index, signal_name in enumerate(active_signal_names):
                        latest_values[signal_name] = float(device_latest_values[index])
                    latest_timestamp_s = max(latest_timestamp_s, device_latest_timestamp_s)
                    last_sample_wallclock_by_device[device_name] = time.time()
                    if device_marked_disconnected[device_name]:
                        print(f"[status] [{resolved_name}] Campioni nuovamente disponibili.")
                        device_marked_disconnected[device_name] = False

            loop_elapsed_s = loop_start - run_start
            joined_values = " | ".join(
                f"{signal_prefix_map[signal_name]}{signal_name}={latest_values[signal_name]}" for signal_name in signal_order
            )
            now_for_print = time.time()
            should_print = self.sample_interval <= 0 or (
                now_for_print - last_print_wallclock
            ) >= self.sample_interval
            if should_print:
                print(
                    f"loop={loop_elapsed_s:.3f}s | data={latest_timestamp_s:.3f}s | {joined_values}"
                )
                if self._udp_sock is not None:
                    payload: dict[str, float] = {f"{signal_prefix_map[name]}{name}": latest_values[name] for name in signal_order}
                    try:
                        self._udp_sock.sendto(
                            json.dumps(payload).encode("utf-8"),
                            (self.udp_host, self.udp_port),
                        )
                    except OSError as exc:
                        print(f"[warn] Invio UDP fallito: {exc}")
                last_print_wallclock = now_for_print

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

        if self._canape_pids_started_by_this_process:
            deadline = time.time() + 3.0
            remaining = set(self._canape_pids_started_by_this_process)

            while remaining and time.time() < deadline:
                time.sleep(0.2)
                remaining &= _get_canape_process_pids()

            for pid in sorted(remaining):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Legge in loop segnali da CANape (E3) su device multipli, stampati a console e inviati via UDP.",
    )
    parser.add_argument(
        "--eth-signal",
        default=[],
        nargs="+",
        help="Uno o piu' nomi segnale Ethernet (es: DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS)",
    )
    parser.add_argument(
        "--eth-signals-file",
        default=str(Path(__file__).resolve().parent / "eth_signals.txt"),
        help="Path file testo segnali Ethernet (uno per riga o separati da spazio; # per commenti)",
    )
    parser.add_argument(
        "--eth-device",
        default="E3_1_2_Premium_HCP1",
        help="Nome device Ethernet in CANape (default: E3_1_2_Premium_HCP1)",
    )
    parser.add_argument(
        "--smotion-signal",
        default=[],
        nargs="+",
        help="Uno o piu' nomi segnale SMOTION (es: AccY_body AccZ_body)",
    )
    parser.add_argument(
        "--smotion-signals-file",
        default=str(Path(__file__).resolve().parent / "smotion_signals.txt"),
        help="Path file testo segnali SMOTION (uno per riga o separati da spazio; # per commenti)",
    )
    parser.add_argument(
        "--smotion-device",
        default="SMOTION",
        help="Nome device SMOTION in CANape (default: SMOTION)",
    )
    parser.add_argument(
        "--signal",
        default=[],
        nargs="+",
        help="Compatibilita': segnali legacy, aggiunti al gruppo Ethernet",
    )
    parser.add_argument(
        "--signals-file",
        default="",
        help="Compatibilita': file segnali legacy, aggiunti al gruppo Ethernet",
    )
    parser.add_argument(
        "--canape-device",
        default="",
        help="Compatibilita' legacy (non usato in modalita' multi-device)",
    )
    parser.add_argument(
        "--canape-folder",
        default=str(_default_canape_folder()),
        help="Cartella progetto CANape usata da Asap3Init5",
    )
    parser.add_argument(
        "--cna-file",
        default="",
        help="Path CNA opzionale",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.01,
        help="Intervallo stampa in secondi (acquisizione sempre alla massima velocita'; default: 0.01)",
    )
    parser.add_argument(
        "--diagnostics",
        "--diagnostic",
        action="store_true",
        help="Stampa diagnostica extra su moduli/device/task",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=2.0,
        help="Secondi tra messaggi di stato quando non arrivano campioni (default: 2)",
    )
    parser.add_argument(
        "--asap-modal",
        dest="asap_modal",
        action="store_true",
        default=True,
        help="Usa modal mode in Asap3Init5 (default: ON)",
    )
    parser.add_argument(
        "--no-asap-modal",
        dest="asap_modal",
        action="store_false",
        help="Disabilita modal mode in Asap3Init5 (fallback)",
    )
    parser.add_argument(
        "--device-discovery-timeout",
        type=float,
        default=6.0,
        help="Secondi massimi per trovare il device dopo init (default: 6)",
    )
    parser.add_argument(
        "--auto-device",
        action="store_true",
        help="Se il device richiesto non viene trovato, usa automaticamente il primo modulo disponibile",
    )
    parser.add_argument(
        "--list-modules-only",
        action="store_true",
        help="Inizializza CANape, stampa i moduli esposti e termina",
    )
    parser.add_argument(
        "--udp-host",
        default="",
        help="IP destinazione UDP per JSON (es: 192.168.1.20). Se omesso, UDP disattivato.",
    )
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5005,
        help="Porta destinazione UDP (default: 5005)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    def _build_signal_list(cli_signals: list[str], file_path: str) -> list[str]:
        signal_names_input = list(cli_signals)
        if file_path:
            signals_path = Path(file_path).resolve()
            if signals_path.exists():
                signal_names_input.extend(_read_signal_names_from_file(signals_path))
            elif not signal_names_input:
                raise SystemExit(f"File segnali non trovato: {signals_path}")
        return _deduplicate_signal_names(signal_names_input)

    eth_signal_names = _build_signal_list(args.eth_signal, args.eth_signals_file)
    smotion_signal_names = _build_signal_list(args.smotion_signal, args.smotion_signals_file)

    legacy_signals = _build_signal_list(args.signal, args.signals_file)
    if legacy_signals:
        eth_signal_names = _deduplicate_signal_names(eth_signal_names + legacy_signals)

    device_signal_configs: list[tuple[str, list[str]]] = []
    if eth_signal_names:
        device_signal_configs.append((args.eth_device, eth_signal_names))
    if smotion_signal_names:
        device_signal_configs.append((args.smotion_device, smotion_signal_names))

    canape_folder = Path(args.canape_folder).resolve()
    cna_file = Path(args.cna_file).resolve() if args.cna_file else None

    try:
        if not device_signal_configs and not args.list_modules_only:
            raise ValueError(
                "Nessun segnale specificato: usa --eth-signals-file/--smotion-signals-file "
                "o --eth-signal/--smotion-signal"
            )

        acq = CanapeE3SignalAcquisition(
            device_signal_configs=device_signal_configs,
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
            acq.shutdown()
            return 0

        total_signals = sum(len(signals) for _, signals in device_signal_configs)
        print(f"Segnali da acquisire (totale={total_signals}):")
        for device_name, signal_names in device_signal_configs:
            print(f"  - {device_name}: {', '.join(signal_names)}")

        try:
            acq.run_loop()
        except KeyboardInterrupt:
            print("\nInterruzione richiesta dall'utente.")
        finally:
            acq.shutdown()

    except Exception as exc:
        print(f"Errore: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())