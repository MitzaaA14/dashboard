#!/usr/bin/env python3
"""Conversie PCD ASCII XYZ în PCD binary XYZ, fără dependențe externe."""
import struct
import sys


def convert(source: str, target: str) -> int:
    points = []
    with open(source, "r", encoding="ascii") as stream:
        for line in stream:
            if line.strip().lower().startswith("data "):
                break
        for line in stream:
            fields = line.split()
            if len(fields) >= 3:
                points.append((float(fields[0]), float(fields[1]), float(fields[2])))
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA binary\n"
    ).encode("ascii")
    with open(target, "wb") as stream:
        stream.write(header)
        for point in points:
            stream.write(struct.pack("<fff", *point))
    return len(points)


if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]))
