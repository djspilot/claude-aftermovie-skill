# GPMF Telemetry — Quick Reference

GoPro's metadata format. The skill reads three streams from it:

- `ACCL` — 3-axis accelerometer at ~200 Hz. Used for jump/impact detection.
- `GYRO` — 3-axis gyroscope at ~400 Hz. Available for steady-footage detection (not currently used in the scorer; would be a good addition).
- `GPS5` — GPS samples at ~18 Hz, each containing `[lat, lon, alt, speed_2d, speed_3d]`. Speed is what the scorer reads.

## Where the data lives

A GoPro MP4 has at least four tracks (`vide`, `soun`, `tmcd`, `meta`). The fifth `meta` track named "GoPro MET" carries the GPMF payload.

ffprobe reveals it:

```bash
ffprobe -i GH010001.MP4 -show_streams 2>&1 | grep -A2 'codec_tag_string=gpmd'
```

Extracting the raw stream:

```bash
ffmpeg -i GH010001.MP4 -map 0:m:handler_name:'GoPro MET' -c copy -f data telemetry.bin
```

(The Python code does the equivalent via stream index lookup.)

## The format

GPMF is a binary nested KLV (Key-Length-Value) format:

```
[4-byte FourCC]  [1-byte type]  [1-byte struct size]  [2-byte repeat count]  [payload]
```

Payload is 32-bit aligned (pad with zeros to the next 4-byte boundary).

Important keys for our purposes:

| Key  | Meaning                                  |
|------|------------------------------------------|
| DEVC | Device container (root)                  |
| STRM | Stream container                         |
| STNM | Human-readable stream name               |
| SCAL | Scale factor — divide raw ints to get real units |
| SIUN | SI units (e.g. "m/s²")                   |
| ACCL | Accelerometer samples (3 axes)           |
| GYRO | Gyroscope samples (3 axes)               |
| GPS5 | GPS lat/lon/alt/speed_2d/speed_3d        |
| TMPC | Camera temperature                       |
| CORI | Camera orientation quaternion (HERO8+)   |
| IORI | Image orientation quaternion             |
| GRAV | Gravity vector                           |

The type byte tells you what numeric type the payload is:
- `b` int8, `B` uint8, `s` int16, `S` uint16, `l` int32, `L` uint32
- `f` float32, `d` float64
- `c` ASCII string
- `\x00` (null) means the payload is itself a nested GPMF structure

## SCAL modifier — easy to get wrong

The `SCAL` key sets a scale factor that applies to subsequent sensor data *in the same nesting level*. If you see:

```
SCAL  s  2  1     [256]
ACCL  s  6  100   [...]
```

then each axis of each ACCL sample should be divided by 256 to get m/s².

The current parser in `aftermovie.py` handles single-value SCAL and 5-value SCAL (for GPS5), which covers HERO5-HERO13. If GoPro adds a new sensor with a more complex scale structure, that parser will need extending.

## HiLight tags (HMMT atom) — separate from GPMF

HiLight tags are NOT in the GPMF stream. They live in the MP4 header's `moov/udta/HMMT` atom:

```
HMMT box:
  uint32 box_size
  4-byte 'HMMT'
  uint32 count
  uint32 timestamp_ms[count]
```

The skill reads these with a direct byte search rather than a full MP4 parser — fragile in theory, robust in practice for GoPro originals.

## When this all stops working

- **Re-encoded clips lose telemetry.** If a user ran the clip through Handbrake or iMovie, the GPMF stream is gone. The analyzer falls back to motion/audio only.
- **GoPro Quik exports lose HiLight tags.** Per GoPro's own docs, clips created or trimmed inside Quik don't preserve HMMT. Use originals from the SD card.
- **HEVC/360 modes from MAX/Fusion** have GPMF but with slightly different stream layout. The current parser handles HERO5-13 reliably; MAX/Fusion may need adjustments.

## References

- Full spec: <https://github.com/gopro/gpmf-parser>
- JavaScript port: <https://github.com/JuanIrache/gopro-telemetry>
- Telemetry Extractor (GUI): <https://goprotelemetryextractor.com>
