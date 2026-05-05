# Windows UDP Server - CANape E3 Multi-device Acquisition

Acquisisce segnali da CANape su due device (Ethernet e SMOTION), stampa i valori a console e opzionalmente invia i dati in UDP come JSON.

## Requisiti

- Windows (usa API ASAP3 via DLL CANape)
- Python 3.9+
- CANape installato sulla macchina

## Setup

```powershell
.\setup_venv.ps1
```

Attivazione manuale (se serve):

```powershell
.\.venvE3\Scripts\Activate.ps1
```

## File segnali

Per default il programma legge:

- eth_signals.txt
- smotion_signals.txt

Formato: un segnale per riga, # per commenti.

Esempio:

```text
# commento
DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS
LWI_AgStgWhl
```

```text
# commento
AccY_body
AccZ_body
```

## Configurazione CANape di default

Cartella progetto passata ad Asap3Init5:

- config/e3/canape/vn5650_smotion

## Avvio rapido

Acquisizione con file segnali di default:

```powershell
python .\main.py
```

Acquisizione + invio UDP:

```powershell
python .\main.py --udp-host 192.168.1.20 --udp-port 5005
```

Esempio payload UDP:

```json
{"t": 1234.5678, "DPM_StDispDrvPosn_XIX_HCP1_15_XIX_VLAN_FAS": 2.0, "LWI_AgStgWhl": 0.12, "AccY_body": -0.05, "AccZ_body": 9.81}
```

## Opzioni principali

- --eth-signal: segnali Ethernet da CLI
- --eth-signals-file: file segnali Ethernet (default: eth_signals.txt)
- --eth-device: device Ethernet CANape (default: E3_1_2_Premium_HCP1)
- --smotion-signal: segnali SMOTION da CLI
- --smotion-signals-file: file segnali SMOTION (default: smotion_signals.txt)
- --smotion-device: device SMOTION CANape (default: SMOTION)
- --canape-folder: cartella progetto CANape (default: config/e3/canape/vn5650_smotion)
- --cna-file: CNA opzionale
- --udp-host: IP destinazione UDP (se omesso, UDP disattivo)
- --udp-port: porta destinazione UDP (default: 5005)
- --sample-interval: intervallo stampa console (default: 0.01)
- --status-interval: intervallo messaggi stato (default: 2.0)
- --list-modules-only: inizializza, stampa moduli, esce
- --auto-device: fallback al primo modulo disponibile

Compatibilita legacy:

- --signal e --signals-file sono ancora supportati e confluiscono nel gruppo Ethernet.
