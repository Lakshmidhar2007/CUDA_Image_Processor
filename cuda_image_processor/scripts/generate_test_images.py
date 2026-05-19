#!/usr/bin/env python3
"""
generate_test_images.py
Generates synthetic PNG images for testing the CUDA image processing pipeline.

Each image contains geometric patterns (gradients, circles, grids, noise)
that make the effects of blur, edge detection, etc. visually obvious.

Usage:
    python3 scripts/generate_test_images.py [--count N] [--output DIR] [--size WxH]
"""

import argparse
import math
import os
import random
import struct
import zlib


def make_png_bytes(rgb_pixels: list, width: int, height: int) -> bytes:
    """
    Minimal pure-Python PNG encoder (no Pillow required).
    rgb_pixels: flat list of (R,G,B) tuples, row-major.
    """
    def png_chunk(name: bytes, data: bytes) -> bytes:
        length = len(data)
        chunk  = name + data
        crc    = zlib.crc32(chunk) & 0xFFFFFFFF
        return struct.pack(">I", length) + chunk + struct.pack(">I", crc)

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # IDAT — build raw scanlines
    raw = b""
    idx = 0
    for y in range(height):
        raw += b"\x00"  # filter type None
        for x in range(width):
            r, g, b = rgb_pixels[idx]; idx += 1
            raw += bytes([r, g, b])

    compressed = zlib.compress(raw, 9)
    idat = png_chunk(b"IDAT", compressed)

    # IEND
    iend = png_chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


def save_png(path: str, pixels: list, width: int, height: int):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(make_png_bytes(pixels, width, height))


# ─── Image generators ────────────────────────────────────────────────────────

def gradient_image(w, h, angle_deg=45):
    """Linear gradient at a given angle."""
    a = math.radians(angle_deg)
    pixels = []
    for y in range(h):
        for x in range(w):
            t = (x * math.cos(a) + y * math.sin(a)) / (w + h) * 2
            t = max(0.0, min(1.0, t))
            r = int(255 * t)
            g = int(255 * (1 - t))
            b = int(128 + 127 * math.sin(t * math.pi))
            pixels.append((r, g, b))
    return pixels


def circles_image(w, h, n_circles=8):
    """Concentric and random circles on a dark background."""
    pixels = [(20, 20, 20)] * (w * h)
    cx, cy = w // 2, h // 2
    for r in range(10, min(w, h) // 2, min(w, h) // (2 * n_circles)):
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        for y in range(h):
            for x in range(w):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if abs(dist - r) < 2:
                    pixels[y * w + x] = color
    return pixels


def checkerboard_image(w, h, tile=32):
    """Black-and-white checkerboard — great for edge detection."""
    pixels = []
    for y in range(h):
        for x in range(w):
            v = 255 if (x // tile + y // tile) % 2 == 0 else 0
            pixels.append((v, v, v))
    return pixels


def noise_image(w, h):
    """Random colour noise."""
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(w * h)]


def stripes_image(w, h, freq=20, direction="horizontal"):
    """Alternating colour stripes."""
    pixels = []
    for y in range(h):
        for x in range(w):
            t = (x if direction == "vertical" else y) // freq % 2
            pixels.append((255, 100, 0) if t == 0 else (0, 100, 255))
    return pixels


def sine_wave_image(w, h, freq=5):
    """Sine-wave pattern — interesting for blur & edge demos."""
    pixels = []
    for y in range(h):
        for x in range(w):
            val = 0.5 + 0.5 * math.sin(2 * math.pi * freq * x / w) \
                      * math.cos(2 * math.pi * freq * y / h)
            r = int(255 * val)
            g = int(255 * (1 - val))
            b = 128
            pixels.append((r, g, b))
    return pixels


def radial_gradient_image(w, h):
    """Radial gradient from centre."""
    cx, cy = w / 2, h / 2
    max_d  = math.sqrt(cx ** 2 + cy ** 2)
    pixels = []
    for y in range(h):
        for x in range(w):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_d
            r = int(255 * (1 - d))
            g = int(128 * d)
            b = int(255 * d)
            pixels.append((r, g, b))
    return pixels


def diagonal_lines_image(w, h, spacing=16):
    """Diagonal lines — emphasises Sobel in both directions."""
    pixels = []
    for y in range(h):
        for x in range(w):
            v = 255 if (x + y) % spacing < 2 else 30
            pixels.append((v, v // 2, 255 - v))
    return pixels


# Map of generator functions
GENERATORS = [
    ("gradient_45",   lambda w, h: gradient_image(w, h, 45)),
    ("gradient_135",  lambda w, h: gradient_image(w, h, 135)),
    ("circles",       circles_image),
    ("checkerboard",  checkerboard_image),
    ("noise",         noise_image),
    ("stripes_h",     lambda w, h: stripes_image(w, h, direction="horizontal")),
    ("stripes_v",     lambda w, h: stripes_image(w, h, direction="vertical")),
    ("sine_wave",     sine_wave_image),
    ("radial",        radial_gradient_image),
    ("diagonal",      diagonal_lines_image),
]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PNG test images")
    parser.add_argument("--count",  type=int, default=20,          help="Number of images to generate")
    parser.add_argument("--output", type=str, default="data/input", help="Output directory")
    parser.add_argument("--size",   type=str, default="512x512",   help="Image size WxH")
    args = parser.parse_args()

    w, h = map(int, args.size.lower().split("x"))
    os.makedirs(args.output, exist_ok=True)

    print(f"Generating {args.count} images ({w}x{h}) in '{args.output}'...")

    random.seed(42)
    for i in range(args.count):
        gen_name, gen_fn = GENERATORS[i % len(GENERATORS)]
        filename = f"img_{i:03d}_{gen_name}.png"
        filepath = os.path.join(args.output, filename)

        pixels = gen_fn(w, h)
        save_png(filepath, pixels, w, h)
        print(f"  [{i+1:3d}/{args.count}] {filename}")

    print(f"\nDone. {args.count} images written to '{args.output}'.")


if __name__ == "__main__":
    main()
