"""Check the demo's actual CSS palette against its 7:1 text contrast target.

Run: venv/bin/python scripts/benchmark/check_demo_contrast.py
Tint/border hues stay unchanged; only foreground text may be lightened.
"""
from pathlib import Path
import re

CSS = Path(__file__).resolve().parents[2] / 'static' / 'command-intel.css'
PALETTE = dict(re.findall(r'(--[\w-]+):\s*(#[0-9a-fA-F]{6})', CSS.read_text().split('\n* {', 1)[0]))


def rgb(color):
    return [int(color[index:index + 2], 16) / 255 for index in (1, 3, 5)]


def mix(foreground, background, amount):
    return [f * amount + b * (1 - amount) for f, b in zip(foreground, background)]


def luminance(color):
    linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in color]
    return sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))


def contrast(foreground, background):
    low, high = sorted((luminance(foreground), luminance(background)))
    return (high + .05) / (low + .05)


def check_palette():
    results = []
    panel = rgb(PALETTE['--panel'])
    for kind in ('executable', 'flag', 'argument', 'operator', 'url', 'ip', 'file', 'registry'):
        hue = rgb(PALETTE['--chip-' + kind])
        text = rgb(PALETTE.get('--chip-' + kind + '-text', PALETTE['--chip-' + kind]))
        background = mix(hue, panel, .18)
        results.append((kind + ' chip', contrast(hue, background), contrast(text, background)))
    for kind in ('red', 'amber', 'green'):
        hue = rgb(PALETTE['--' + kind])
        text = rgb(PALETTE.get('--' + kind + '-text', PALETTE['--' + kind]))
        for name, tint in (('verdict', .1), ('indicator', .18)):
            background = mix(hue, panel, tint)
            results.append((kind + ' ' + name, contrast(hue, background), contrast(text, background)))
    for background in ('bg', 'panel', 'raised'):
        for text in ('text', 'secondary'):
            value = contrast(rgb(PALETTE['--' + text]), rgb(PALETTE['--' + background]))
            results.append((text + ' on ' + background, value, value))
    for background in ('accent', 'violet'):
        value = contrast(rgb(PALETTE['--bg']), rgb(PALETTE['--' + background]))
        results.append(('button on ' + background, value, value))
    for name, original, final in results:
        print(f'{name:24} requested {original:5.2f}:1 -> final {final:5.2f}:1')
    failures = [(name, final) for name, _, final in results if final < 7]
    if failures:
        raise SystemExit(f'Contrast failed: {failures}')
    return results


if __name__ == '__main__':
    check_palette()
