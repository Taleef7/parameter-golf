#!/usr/bin/env python3
"""Extract val_bpb from a training log file.

Usage: python experiments/verify_run.py <log_file_path>

Searches for 'final_int8_zlib_roundtrip val_bpb:<value>' in the log.
Prints 'val_bpb: <value>' on success, exits 1 if not found.
"""
import re
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_run.py <log_file_path>", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    try:
        with open(log_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    match = re.search(r"final_int8_zlib_roundtrip val_bpb:([0-9.]+)", content)
    if not match:
        print("Error: 'final_int8_zlib_roundtrip val_bpb:' not found in log", file=sys.stderr)
        sys.exit(1)

    print(f"val_bpb: {match.group(1)}")


if __name__ == "__main__":
    main()
