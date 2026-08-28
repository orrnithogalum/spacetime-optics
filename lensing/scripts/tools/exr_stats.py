#!/usr/bin/env python3
"""Luminance histogram of an HDR frame written by lensim (output.exr = 1). Standard library only.

    python3 scripts/tools/exr_stats.py out/scene/hdr/frame_00000_ideal.exr
"""
import array
import math
import struct
import sys


def read_lensim_exr(path):
    d = open(path, "rb").read()
    pos, attrs = 8, {}
    while d[pos] != 0:
        e = d.index(b"\0", pos)
        name = d[pos:e].decode()
        e2 = d.index(b"\0", e + 1)
        size = struct.unpack_from("<i", d, e2 + 1)[0]
        attrs[name] = d[e2 + 5:e2 + 5 + size]
        pos = e2 + 5 + size
    pos += 1
    x0, y0, x1, y1 = struct.unpack("<4i", attrs["dataWindow"])
    W, H = x1 - x0 + 1, y1 - y0 + 1
    offsets = struct.unpack_from("<%dQ" % H, d, pos)
    lum = []
    for y in range(H):
        row = array.array("f")
        row.frombytes(d[offsets[y] + 8:offsets[y] + 8 + W * 12])
        B, G, R = row[0:W], row[W:2 * W], row[2 * W:]
        lum.extend(0.2126 * R[x] + 0.7152 * G[x] + 0.0722 * B[x] for x in range(W))
    return W, H, lum


def main():
    W, H, lum = read_lensim_exr(sys.argv[1])
    nz = sorted(v for v in lum if v > 0)
    print("%dx%d, %d non-zero pixels, max %.4g cd/m^2" % (W, H, len(nz), nz[-1] if nz else 0))
    hist = {}
    for v in nz:
        k = int(math.floor(math.log10(v)))
        hist[k] = hist.get(k, 0) + 1
    for k in sorted(hist):
        print("  1e%+03d .. 1e%+03d : %8d" % (k, k + 1, hist[k]))
    for p in (0.5, 0.9, 0.95, 0.99, 0.995):
        if nz:
            print("  p%-5g = %.4g" % (p * 100, nz[int(p * (len(nz) - 1))]))


if __name__ == "__main__":
    main()
