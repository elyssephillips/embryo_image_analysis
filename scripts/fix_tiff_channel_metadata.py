"""Re-save TIFFs with correct ImageJ channel metadata so napari reads them as
separate channels instead of a flat Z*C stack.

Assumes pages were written in channel-major order (all Z of ch0, then ch1...),
which is the order produced by convert_h5_channels_to_tiff.py.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tifffile

from src.conversion import (
    get_config_value,
    get_h5_conversion_config,
    load_pipeline_raw_dir,
    load_yaml_config,
    resolve_relative_path,
)


def fix_tiff(path: Path, n_channels: int) -> None:
    data = tifffile.imread(path)
    print(f"  {path.name}: read shape {data.shape}, ndim={data.ndim}")
    if data.ndim == 4:
        # Already shaped — could be (C,Z,Y,X) or (Z,C,Y,X); normalise to (C,Z,Y,X)
        if data.shape[0] == n_channels:
            pass  # already (C,Z,Y,X)
        elif data.shape[1] == n_channels:
            data = data.transpose(1, 0, 2, 3)  # (Z,C,Y,X) → (C,Z,Y,X)
        else:
            print(f"  SKIP {path.name}: 4D shape {data.shape} but neither axis matches n_channels={n_channels}")
            return
    elif data.ndim == 3:
        if data.shape[0] % n_channels != 0:
            print(f"  SKIP {path.name}: {data.shape[0]} pages not divisible by n_channels={n_channels}")
            return
        data = data.reshape(n_channels, -1, *data.shape[1:])  # (C,Z,Y,X)
    else:
        print(f"  SKIP {path.name}: unexpected ndim={data.ndim}, shape={data.shape}")
        return
    n_ch, nz, ny, nx = data.shape
    imagej_desc = (
        "ImageJ=1.11a\n"
        f"images={n_ch * nz}\n"
        f"channels={n_ch}\n"
        f"slices={nz}\n"
        "hyperstack=true\n"
        "mode=composite\n"
    )
    # Transpose to (Z, C, Y, X) so pages are in TZC order (z0c0, z0c1, z1c0, ...)
    # which is what ImageJ expects when channels > 1.
    data_zcyx = data.transpose(1, 0, 2, 3)
    tifffile.imwrite(path, data_zcyx, description=imagej_desc, contiguous=True)
    print(f"  Fixed {path.name}: shape now {data.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix ImageJ channel metadata in existing TIFFs so napari opens them correctly."
    )
    parser.add_argument("tiff_dir", type=Path, nargs="?", help="Directory containing TIFFs to fix.")
    parser.add_argument("--n-channels", type=int, default=None, help="Number of channels per TIFF.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file.")
    parser.add_argument("--pipeline-config", type=Path, default=None, help="Pipeline YAML config.")
    args = parser.parse_args()

    default_config_path = None
    for candidate in [Path("configs/config.yaml"), Path("config.yaml")]:
        if candidate.exists():
            default_config_path = candidate
            break
    if default_config_path is None:
        found = list(Path("configs").glob("**/config.yaml")) if Path("configs").exists() else []
        if len(found) == 1:
            default_config_path = found[0]

    config_path = args.config or default_config_path
    config = load_yaml_config(config_path) if config_path else None
    config_base = config_path.parent if config_path else Path.cwd()
    h5_config = get_h5_conversion_config(config)

    pipeline_config_path = (
        args.pipeline_config
        or get_config_value(h5_config, ["pipeline_config"])
        or get_config_value(config, ["pipeline_config"])
    )
    if pipeline_config_path is not None:
        pipeline_config_path = resolve_relative_path(config_base, pipeline_config_path)

    tiff_dir = args.tiff_dir
    if tiff_dir is None:
        raw = (
            get_config_value(h5_config, ["output_dir"])
            or get_config_value(config, ["raw_data_dir"])
            or (load_pipeline_raw_dir(pipeline_config_path) if pipeline_config_path else None)
        )
        if raw is None:
            print("Error: provide tiff_dir or a config with raw_data_dir/output_dir.")
            sys.exit(1)
        tiff_dir = Path(raw)

    n_channels = args.n_channels
    if n_channels is None:
        channel_names = get_config_value(config, ["microscopy", "channel_names"])
        if channel_names:
            n_channels = len(channel_names)
    if n_channels is None:
        print("Error: --n-channels is required (or set microscopy.channel_names in the config).")
        sys.exit(1)

    tiffs = sorted(tiff_dir.glob("*.tif")) + sorted(tiff_dir.glob("*.tiff"))
    if not tiffs:
        print(f"No TIFFs found in {tiff_dir}")
        sys.exit(0)

    print(f"Fixing {len(tiffs)} TIFF(s) in {tiff_dir} (n_channels={n_channels})...")
    for path in tiffs:
        fix_tiff(path, n_channels)
    print("Done.")


if __name__ == "__main__":
    main()
