# Windows UDP Server — CANape E3 Signal Acquisition

Reads one or more signals from CANape (E3 interface) in a loop, prints them to the console, and optionally streams each sample as a JSON packet over UDP.

---

## Requirements

- **Windows only** — uses the CANape ASAP3 API DLL
- **Python 3.9+**
- CANape installed on the machine (the DLL is auto-discovered from the installation)

No external Python packages are required — only the standard library is used.

---

## Setup

```powershell
.\setup_venv.ps1
```

This creates a `.venvE3` virtual environment and installs dependencies from `requirements.txt`.

Activate manually if needed:

```powershell
.\.venvE3\Scripts\Activate.ps1
```

---

## Signal List

Signals to acquire are listed in `signals.txt`, one per line. Lines starting with `#` are comments.

```
# Example signals.txt
Sig1
Sig2
Sig3
Sig4
```

Signals can also be passed directly on the command line with `--signal` (see below).

---

## Usage

### Basic — read signals defined in `signals.txt`

```powershell
python main.py
```

### Specify signals inline

```powershell
python main.py --signal VehV_v VehAX_ax
```

### Enable UDP output

```powershell
python main.py --udp-host 192.168.1.20 --udp-port 5005
```

Each acquired sample is sent as a JSON packet:

```json
{"t": 1234.5678, "Sig1": 0.0, "Sig2": -1.5, "Sig3": 0.3, "Sig4": 2.1}
```

When no samples are available (FIFO empty), a zero-value packet is sent at `--status-interval` rate:

```json
{"t": 0.0, "Sig1": 0, "Sig2": 0, "Sig3": 0, "Sig4": 0}
```

If `--udp-host` is omitted, UDP output is disabled.

### Discover available CANape modules

```powershell
python main.py --list-modules-only
```

---

## All Options

| Argument | Default | Description |
|---|---|---|
| `--signal` | — | One or more signal names (space-separated) |
| `--signals-file` | `signals.txt` | Path to signal list file |
| `--canape-device` | `e3` | CANape device name |
| `--canape-folder` | `config/e3/canape` | CANape project folder (passed to `Asap3Init5`) |
| `--cna-file` | — | Optional `.cna` project file to load |
| `--sample-interval` | `0.1` | Loop interval in seconds |
| `--status-interval` | `2.0` | Seconds between status messages when FIFO is empty |
| `--device-discovery-timeout` | `6.0` | Max seconds to wait for device after init |
| `--auto-device` | off | Auto-select the first available module if the named device is not found |
| `--diagnostics` | off | Print extra diagnostics on modules/device/task |
| `--asap-modal` / `--no-asap-modal` | on | Enable/disable modal mode in `Asap3Init5` |
| `--list-modules-only` | off | Init CANape, print exposed modules, then exit |
| `--udp-host` | — | UDP destination IP. If omitted, UDP is disabled |
| `--udp-port` | `5005` | UDP destination port |

---

## Console Output Format

Each sample is printed as:

```
1234.568s | Sig1=0.0 | Sig2=-1.5 | Sig3=0.3 | Sig4=2.1
```

When the FIFO is empty:

```
[status] Nessun campione disponibile (FIFO=0) sul task 1
0.000s | Sig1=0 | Sig2=0 | Sig3=0 | Sig4=0
```

Press **CTRL+C** to stop acquisition cleanly.
