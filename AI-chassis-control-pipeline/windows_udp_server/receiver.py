from __future__ import annotations

import argparse
from datetime import datetime
import json
import socket
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Riceve pacchetti UDP JSON e li stampa a terminale.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="IP locale su cui mettersi in ascolto (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Porta UDP locale (default: 5005)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=65535,
        help="Dimensione buffer recvfrom (default: 65535)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))

    print(f"[receiver] In ascolto su {args.host}:{args.port} (CTRL+C per uscire)")

    try:
        while True:
            data, addr = sock.recvfrom(args.buffer_size)
            raw_text = data.decode("utf-8", errors="replace")
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Se e' JSON valido, lo ristampa ordinato; altrimenti stampa il testo raw.
            try:
                parsed = json.loads(raw_text)
                pretty = json.dumps(parsed, ensure_ascii=False, separators=(",", ": "))
                print(f"[{timestamp}] {addr[0]}:{addr[1]} -> {pretty}")
            except json.JSONDecodeError:
                print(f"[{timestamp}] {addr[0]}:{addr[1]} -> {raw_text}")

    except KeyboardInterrupt:
        print("\n[receiver] Terminazione richiesta.")
    finally:
        sock.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
