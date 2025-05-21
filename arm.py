#!/usr/bin/env python3
import argparse
import json
import socket
import struct
import sys

# Adjust this to your VM’s host-only IP and port
GUEST_IP = "192.168.20.124"
GUEST_PORT = 5000

def send_command(cmd: dict):
    payload = json.dumps(cmd).encode("utf-8")
    header  = struct.pack("!I", len(payload))
    with socket.create_connection((GUEST_IP, GUEST_PORT), timeout=2) as sock:
        sock.sendall(header + payload)

def main():
    p = argparse.ArgumentParser(description="Arm network client")
    sub = p.add_subparsers(dest="op", required=True)

    m1 = sub.add_parser("move1", help="Move a single servo")
    m1.add_argument("--id",    type=int,   required=True, help="Servo ID (1–6)")
    m1.add_argument("--angle", type=int,   required=True, help="Angle (0–240)")
    m1.add_argument("--time",  type=int,   default=1000,   help="Duration ms")

    m6 = sub.add_parser("move6", help="Move all six servos")
    m6.add_argument(
        "--pos", nargs=6, type=int, required=True,
        metavar=("J1","J2","J3","J4","J5","J6"),
        help="Six angles in degrees"
    )
    m6.add_argument("--time", type=int, default=1500, help="Duration ms")

    args = p.parse_args()
    if args.op == "move1":
        cmd = {"op":"move1", "id":args.id, "angle":args.angle, "time":args.time}
    else:
        cmd = {"op":"move6", "pos":args.pos, "time":args.time}

    try:
        send_command(cmd)
        print("Sent:", cmd)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
