# Genos demo color and platform verification

The Base64 identity sample (`whoami /all && net localgroup administrators`) now returns `windows` from the parser and shows `windows` in the UI. Bare `whoami` returns `unknown` because the binary name alone does not identify an OS.

Colors are centralized in `static/command-intel.css`. Token backgrounds retain the requested 18% tint and borders retain the 55% hue. Foregrounds were lightened where necessary. Existing layout, sample text, motion choreography and API display remain intact. Reduced-motion and increased-contrast preferences were checked in Chromium.

## Contrast corrections

All entries below failed the 7:1 target; red indicator text also failed 4.5:1. Background tints were preserved.

| Pair | Requested ratio | Final text | Final ratio |
| --- | ---: | --- | ---: |
| Blue flag on its tint | 5.08:1 | `#A9CBFF` | 8.35:1 |
| Magenta URL on its tint | 5.10:1 | `#FF9BE8` | 7.70:1 |
| Orange IP on its tint | 5.48:1 | `#FFB18C` | 8.05:1 |
| Violet file / semantic tag on its tint | 5.10:1 | `#CAB8FF` | 7.81:1 |
| Red verdict on its wash | 4.87:1 | `#FF91A2` | 7.93:1 |
| Red IOC chip / count badge on its tint | 4.43:1 | `#FF91A2` | 7.21:1 |

273 computed text pairs passed the Chromium audit. Minimum observed ratio: 7.11:1. The Analyze gradient was checked against both color endpoints. A dependency-free checker is available at `scripts/benchmark/check_demo_contrast.py`; computed audit results are in `color-contrast-audit.json`. The empty-state legend and disabled Analyze button retain full contrast.

## Actual measurements

NVIDIA GeForce RTX 4060, two successive API scans per sample after engine startup warmup. Displayed milliseconds came directly from each response, with no animation delay included. VRAM is peak additional PyTorch allocation above the resident baseline, not total device memory.

| Sample | Run | Analysis time | Command VRAM |
| --- | ---: | ---: | ---: |
| identity | 1 | 63.1 ms | 376.3 MiB |
| identity | 2 | 50.6 ms | 376.3 MiB |
| retrieval | 1 | 59.9 ms | 376.3 MiB |
| retrieval | 2 | 66.9 ms | 376.3 MiB |

Use **376.3 MiB additional GPU allocation per command** for these captured demo results. The measured times ranged from **50.6 to 66.9 ms**, rather than 14 ms. Runtime figures can vary between calls; screenshots and response files contain the corresponding actual values. No slide draft was present in the workspace to update.

## Verification

19 regression tests passed (5 platform, 3 indicator, 7 GPU telemetry, 4 MITRE ranking). Chromium checks passed for the real API's Windows platform, top-five MITRE, syntax token types, linked hover glow, animation skip, 24px VRAM row, 1920px/1440px/768px/375px layouts, reduced motion and increased contrast. There were no browser errors.

Screenshots: `color-base64.png`, `color-multi-indicator.png`, and `color-multi-indicator-indicators.png`.

The top bar and LIVE DEMO label remain removed as previously requested; the existing accent dot uses cyan. No other requested feature was cut.
