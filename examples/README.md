# SatSim Examples

## Configs

- **`basic.yaml`** — Minimal working example. Fixed parameters, 3 satellite targets, 8 frames. Good starting point for understanding the config format.
- **`randomized.yaml`** — Randomized scene generation using `$sample`. Demonstrates per-pixel non-uniformity, random PSF, random exposure times, and random target populations.

## Running

From the command line:

```bash
# basic example
satsim run examples/basic.yaml -o output/basic/

# randomized with seed for reproducibility
satsim run examples/randomized.yaml -o output/randomized/ --seed 42

# with debug logging to see what's happening
satsim --debug INFO run examples/basic.yaml -o output/basic/
```

From Python:

```python
import satsim

# iterate over frames
config = satsim.load_yaml('examples/basic.yaml')
for frame in satsim.generate(config, seed=42):
    image = frame.fpa_digital.numpy()
    print(f"Frame {frame.frame_num}: shape={image.shape}, "
          f"min={image.min():.0f}, max={image.max():.0f}")

# or write directly to disk
satsim.run('examples/basic.yaml', output_dir='output/basic/', seed=42)
```

## Config Reference

See `schema/v1/Document.json` for the full configuration schema. Key sections:

| Section | Purpose |
|---------|---------|
| `sim` | Convolution mode, oversampling factors, padding |
| `fpa` | Sensor: size, FOV, noise, PSF, exposure, A/D conversion |
| `background` | Sky background and stray light |
| `geometry.stars` | Star field generation (bins or catalog) |
| `geometry.obs` | Satellite targets (static list, generators, TLE orbits) |
| `geometry.site` | Observer location for topocentric simulations |

### Dynamic Keywords

Any numeric value can be replaced with a `$sample` to randomize it:

```yaml
# fixed value
eod: 0.3

# randomized per scene realization
eod:
  $sample: random.uniform
  low: 0.15
  high: 0.6

# deterministic random (same value every run)
eod:
  $sample: random.uniform
  low: 0.15
  high: 0.6
  seed: 42
```

Other keywords: `$ref` (cross-references), `$generator` (Python functions), `$function`, `$compound` (arithmetic), `$import` (external files).
