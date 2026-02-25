"""Simplified CLI for SatSim v2.0.

Replaces TensorFlow device management with PyTorch device selection.
"""

import sys
import os
import logging
import math

import click

from satsim._version import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name='SatSim')
@click.option('-d', '--debug', default='WARNING', show_default=True,
              help='Set the logging level. [DEBUG,INFO,WARNING,ERROR,OFF]')
@click.pass_context
def main(ctx, debug):
    """Command line interface (CLI) for SatSim."""
    logging.basicConfig(format='%(asctime)s %(name)-20s %(levelname)-8s %(message)s')
    if debug == 'OFF':
        logging.getLogger().propagate = False
    elif debug == 'DEBUG':
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.DEBUG)
    elif debug == 'INFO':
        logging.getLogger().setLevel(logging.INFO)
    elif debug == 'ERROR':
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    return 0


@main.command()
@click.pass_context
def version(ctx):
    """Print the version."""
    print(__version__)
    return 0


@main.command(help='Run simulation from configuration file.')
@click.option('-d', '--device', default='auto', type=str,
              help='Device: auto, cpu, cuda, cuda:0, cuda:1, etc.')
@click.option('-r', '--memory', default=None, type=int,
              help='GPU maximum memory limit in megabytes.')
@click.option('-s', '--seed', default=None, type=int,
              help='Random seed for reproducibility.')
@click.option('-o', '--output_dir', default='./', help='Output directory.')
@click.option('-i', '--output_intermediate', is_flag=True,
              help='Output intermediate debug files.')
@click.argument('config_file', required=True)
@click.pass_context
def run(ctx, device, memory, seed, output_dir, config_file, output_intermediate):
    """Run a simulation from a config file."""
    from satsim.device import DeviceConfig, set_device

    # Configure device
    if device == 'auto':
        dc = DeviceConfig(device_type='auto', memory_limit_mb=memory)
    elif device == 'cpu':
        dc = DeviceConfig(device_type='cpu')
    elif device.startswith('cuda'):
        parts = device.split(':')
        device_id = int(parts[1]) if len(parts) > 1 else 0
        dc = DeviceConfig(device_type='cuda', device_id=device_id, memory_limit_mb=memory)
    else:
        logger.error('Unknown device: %s', device)
        sys.exit(1)

    resolved_device = dc.resolve()
    set_device(resolved_device)

    logger.info('SatSim version %s.', __version__)
    logger.info('Device: %s', resolved_device)

    # Load config
    from satsim.config.loading import load_yaml, load_json

    if config_file.endswith(('.yml', '.yaml')):
        logger.info('Loading YAML file: %s', config_file)
        ssp = load_yaml(config_file)
    elif config_file.endswith('.json'):
        logger.info('Loading JSON file: %s', config_file)
        ssp = load_json(config_file)
    else:
        logger.error('File type unknown. Config file must be .json or .yml.')
        sys.exit(1)

    # Dispatch to radar simulator if radar config detected
    if isinstance(ssp, dict) and 'radar' in ssp:
        from satsim.radar import simulate_from_file
        logger.info('Detected radar configuration. Dispatching to radar simulator.')
        out_dir = simulate_from_file(config_file, output_dir)
        logger.info('Saved radar observations to: %s', out_dir)
        return 0

    # Store input directory for relative path resolution
    ssp['_input_dir'] = os.path.dirname(os.path.abspath(config_file))

    # Run EO/optical pipeline
    from satsim.io.writers import write_collection
    dir_name = write_collection(ssp, output_dir, seed=seed)
    logger.info('Output saved to: %s', dir_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
