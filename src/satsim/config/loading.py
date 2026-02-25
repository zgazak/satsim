"""Config loading and realization.

Loads YAML/JSON configs and resolves Sampled values.
Also provides backward compatibility with existing JSON DSL configs.
"""

from __future__ import annotations

import json
import copy
import os
import logging
import collections.abc
import importlib
import functools
import operator
import pickle

import numpy as np

from satsim.math.random import gen_sample
from satsim.config.seeding import SeedTree

logger = logging.getLogger(__name__)

_config = {}
_SEED_MAX = 2 ** 32


def load_json(filename: str) -> dict:
    """Load a JSON config file.

    Args:
        filename: Path to JSON file.

    Returns:
        Config as a dictionary.
    """
    with open(filename, 'r') as f:
        return json.load(f)


def load_yaml(filename: str) -> dict:
    """Load a YAML config file.

    Args:
        filename: Path to YAML file.

    Returns:
        Config as a dictionary.
    """
    import yaml
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def realize(config: dict, seed: int | None = None) -> dict:
    """Resolve all dynamic keywords in a config dict.

    Replaces the multi-pass transform() engine. Processes $sample, $ref,
    $import, $generator, $function, $compound keywords.

    Args:
        config: Raw config dictionary.
        seed: Optional root seed for deterministic sampling.

    Returns:
        Fully resolved config dictionary.
    """
    config = copy.deepcopy(config)
    dirname = config.get('_input_dir', None)

    global _config
    _config = config

    # Apply seed tree for deterministic sampling
    if seed is not None:
        _apply_seed_tree(config, np.random.RandomState(seed))

    # Phase 1: resolve $sample
    for _ in range(5):
        config = _transform(config, dirname, False)
        if not _has_rkey_deep(config, '$sample'):
            break

    # Phase 2: resolve $import
    for _ in range(20):
        config = _import({'root': config})['root']
        if not _has_rkey_deep(config, '$import'):
            break

    # Phase 3: resolve $sample again (from imports)
    for _ in range(5):
        config = _transform(config, dirname, False)
        if not _has_rkey_deep(config, '$sample'):
            break

    # Phase 4: resolve $ref
    for _ in range(5):
        config = _ref(config, config)
        if not _has_rkey_deep(config, '$ref'):
            break

    # Phase 5: resolve $sample again (from refs)
    for _ in range(5):
        config = _transform(config, dirname, False)
        if not _has_rkey_deep(config, '$sample'):
            break

    # Phase 6: resolve $generator and $function
    for _ in range(5):
        config = _transform(config, dirname, True)
        config = _transform(config, dirname, False)
        if (not _has_rkey_deep(config, '$sample') and
            not _has_rkey_deep(config, '$generator') and
            not has_key_deep(config, '$function')):
            break

    # Phase 7: resolve $compound
    for _ in range(5):
        if not _has_rkey_deep(config, '$compound'):
            break
        config = _transform(config, dirname, run_compound=True)

    return config


# Backward-compatible transform() for existing code
def transform(config, dirname=None, max_stages=5, with_debug=False, max_imports=20, max_ref=5):
    """Transform a SatSim configuration, evaluating dynamic keywords.

    This is the backward-compatible version of realize().
    """
    stages = []
    global _config
    _config = config

    sim_param = config.get('sim', {})
    ver_param = config.get('version', '1.0')

    def eval_sample(config):
        for i in range(max_stages):
            config = _transform(config, dirname, False)
            stages.append(copy.deepcopy(config))
            if not _has_rkey_deep(config, '$sample'):
                break
        return config

    config = eval_sample(config)

    for i in range(max_imports):
        config = _import({'root': config})['root']
        if not _has_rkey_deep(config, '$import'):
            break

    config['sim'] = sim_param
    config['version'] = ver_param
    stages.append(copy.deepcopy(config))

    config = eval_sample(config)

    for i in range(max_ref):
        config = _ref(config, config)
        if not _has_rkey_deep(config, '$ref'):
            break

    stages.append(copy.deepcopy(config))
    config = eval_sample(config)

    for i in range(max_stages):
        config = _transform(config, dirname, True)
        stages.append(copy.deepcopy(config))
        config = _transform(config, dirname, False)
        stages.append(copy.deepcopy(config))
        if (not _has_rkey_deep(config, '$sample') and
            not _has_rkey_deep(config, '$generator') and
            not has_key_deep(config, '$function')):
            break

    for i in range(max_stages):
        if not _has_rkey_deep(config, '$compound'):
            break
        config = _transform(config, dirname, run_compound=True)
        stages.append(copy.deepcopy(config))

    if with_debug:
        return config, stages
    else:
        return config


def save_debug(configs, output_dir):
    """Write config stages to files for debugging."""
    if output_dir is not None:
        for config, stage in zip(configs, range(len(configs))):
            with open(os.path.join(output_dir, 'config_pass_{}.json'.format(stage + 1)), 'w') as json_file:
                json.dump(config, json_file, indent=4, default=lambda o: "n/a")


def save_json(filename, config, save_pickle=False):
    """Save a configuration to JSON."""
    def serialize(o):
        if save_pickle:
            picklename = '{:05d}.pickle'.format(serialize.i)
            with open(os.path.join(os.path.dirname(filename), picklename), 'wb') as picklefile:
                pickle.dump(o, picklefile)
        else:
            picklename = 'n/a'
        serialize.i = serialize.i + 1
        return {'$file': picklename}

    serialize.i = 0

    with open(filename, 'w') as outfile:
        json.dump(config, outfile, indent=4, default=serialize)


def save_cache(param, value):
    """Save a parameter to cache (no-op in v2, cache removed)."""
    pass


# --- Internal helpers (ported from old config.py) ---

def _resolve_seed(param):
    if 'seed' not in param:
        return None
    seed = param['seed']
    if seed is None:
        return None
    if isinstance(seed, (dict, list)):
        seed_param = {'seed': seed}
        seed_param = _ref(seed_param, _config)
        seed_param = _transform(seed_param)
        seed = seed_param['seed']
    return seed


def _apply_seed_tree(param, rng):
    if isinstance(param, dict):
        if _has_rkey(param, '$sample'):
            if 'seed' not in param or param['seed'] is None:
                param['seed'] = int(rng.randint(0, _SEED_MAX))
        for k, v in param.items():
            if k == 'seed':
                continue
            if isinstance(v, (dict, list)):
                _apply_seed_tree(v, rng)
    elif isinstance(param, list):
        for v in param:
            if isinstance(v, (dict, list)):
                _apply_seed_tree(v, rng)
    return param


def parse_function(param):
    """Parse a function spec into a callable."""
    module = importlib.import_module(param['module'])
    if '$function' in param:
        function = getattr(module, param['$function'])
    else:
        function = getattr(module, param['function'])

    if 'kwargs' in param:
        kwargs2 = param['kwargs']
        def f(*args, **kwargs):
            kwargs2.update(**kwargs)
            return function(*args, **kwargs2)
        return f
    else:
        return function


def parse_function_pipeline(param):
    """Parse a pipeline of functions."""
    ff = list(map(parse_function, param))
    def func(x, t=None, **kwargs):
        for f in ff:
            try:
                from inspect import signature
                sig = signature(f)
                if 't' in sig.parameters:
                    x = f(x, t, **kwargs)
                else:
                    x = f(x, **kwargs)
            except Exception:
                x = f(x, t, **kwargs)
        return x
    return func


def parse_import(param):
    """Parse an $import directive."""
    with open(param[_rkey(param, '$import')]) as json_file:
        data = json.load(json_file)
        if 'key' in param:
            keys = param['key'].split('.')
            data = functools.reduce(operator.getitem, keys, data)
        if 'override' in param:
            dict_merge(data, param['override'])
        return data


def parse_generator(param):
    """Parse and run a generator function."""
    f = parse_function(param)
    return f()


def parse_random_sample(param):
    """Parse a $sample parameter and return a sample."""
    compat_name = _rkey(param, '$sample')
    stype, rtype = param[compat_name].split('.')
    seed = _resolve_seed(param)

    if stype == 'random':
        if rtype == 'choice':
            rng = np.random.RandomState(seed)
            c = param['choices']
            choice = copy.deepcopy(c[rng.randint(0, len(c))])
            if seed is not None:
                _apply_seed_tree(choice, rng)
            return _transform(parse_param(choice))
        elif rtype == 'list':
            samples = []
            rng = None
            if seed is not None:
                rng = np.random.RandomState(seed)
                length_param = _apply_seed_tree(copy.deepcopy(param['length']), rng)
                list_length = parse_param(length_param)
            else:
                list_length = parse_param(param['length'])
            if 'list' in param:
                logger.warning('Deprecated list replacement. Use $sample keyword inline instead.')
                for i in range(list_length):
                    value = copy.deepcopy(param['list'])
                    if seed is not None:
                        _apply_seed_tree(value, rng)
                    samples.append(_transform(parse_param(value)))
                param['list'] = samples
                del param[compat_name]
                del param['length']
            else:
                for i in range(list_length):
                    value = copy.deepcopy(param['value'])
                    if seed is not None:
                        _apply_seed_tree(value, rng)
                    samples.append(_transform(parse_param(value)))
                del param[compat_name]
                param = samples
            return param
        else:
            if seed is not None:
                _apply_seed_tree(param, np.random.RandomState(seed))
            del param[compat_name]
            param = _ref(param, _config)
            param = _transform(param)
            return gen_sample(rtype, **param)


def _normalize_compound_operator(op):
    if op is None:
        return 'add'
    op_norm = str(op).strip().lower()
    if op_norm in {'add', '+', 'sum'}:
        return 'add'
    if op_norm in {'multiply', 'mul', '*', 'product'}:
        return 'multiply'
    raise ValueError('Unsupported compound operator: {}'.format(op))


def _evaluate_compound(param, dirname, eval_python):
    compound_key = _rkey(param, '$compound')
    operator_key = _rkey(param, '$operator') if _has_rkey(param, '$operator') else None
    default_op = _normalize_compound_operator(param[operator_key] if operator_key else 'add')

    items = param[compound_key]
    if len(items) == 0:
        return items

    acc = None
    for item in items:
        item_op = default_op
        item_value = item

        if isinstance(item_value, dict) and _has_rkey(item_value, '$operator'):
            item_op_key = _rkey(item_value, '$operator')
            item_op = _normalize_compound_operator(item_value[item_op_key])
            item_value = copy.deepcopy(item_value)
            del item_value[item_op_key]

        item_value = parse_param(item_value, dirname, run_generator=True, eval_python=eval_python, run_compound=False)
        item_value = _transform(item_value, dirname, run_generator=True, eval_python=eval_python, run_compound=False)

        if acc is None:
            acc = item_value
            continue

        if item_op == 'add':
            acc = acc + item_value
        elif item_op == 'multiply':
            acc = acc * item_value
        else:
            raise ValueError('Unsupported compound operator: {}'.format(item_op))

    return acc


def parse_param(param, dirname=None, run_generator=False, eval_python=False, run_compound=False):
    """Parse a parameter and recursively parse children."""
    if isinstance(param, dict):
        if _has_rkey(param, '$sample'):
            if _has_rkey(param, '$operator') and not run_compound:
                return param
            return parse_random_sample(param)
        elif _has_rkey(param, '$file'):
            with open(os.path.join(dirname, param[_rkey(param, '$file')]), 'rb') as f:
                return pickle.load(f)
        elif run_generator and '$function' in param:
            if _has_rkey(param, '$operator') and not run_compound:
                return param
            val = parse_function(param)()
            save_cache(param, val)
            return val
        elif run_generator and _has_rkey(param, '$generator'):
            if _has_rkey(param, '$operator') and not run_compound:
                return param
            return parse_generator(param[_rkey(param, '$generator')])
        elif eval_python and _has_rkey(param, '$pipeline'):
            return parse_function_pipeline(param[_rkey(param, '$pipeline')])
        elif run_compound and _has_rkey(param, '$compound'):
            val = _evaluate_compound(param, dirname, eval_python)
            save_cache(param, val)
            return val
        else:
            return param
    elif isinstance(param, list):
        param_out = []
        for i in range(len(param)):
            if isinstance(param[i], list):
                p = parse_param(param[i], dirname, run_generator, eval_python)
                param_out.append(p)
            else:
                p = parse_param(param[i], dirname, run_generator, eval_python)
                if isinstance(p, list):
                    param_out.extend(p)
                else:
                    param_out.append(p)
        return param_out
    else:
        return param


def _ref(config, original):
    """Evaluate $ref references."""
    for k, v in config.items():
        if isinstance(v, dict):
            if _has_rkey(v, '$ref'):
                name = _rkey(v, '$ref')
                keys = v[name].split('.')
                config[k] = functools.reduce(operator.getitem, keys, original)
            else:
                config[k] = _ref(v, original)
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    if _has_rkey(item, '$ref'):
                        name = _rkey(item, '$ref')
                        keys = item[name].split('.')
                        v[idx] = functools.reduce(operator.getitem, keys, original)
                    else:
                        _ref(item, original)
                elif isinstance(item, list):
                    _ref({'_list': item}, original)
    return config


def _import(config):
    """Evaluate $import directives."""
    for k, v in config.items():
        if isinstance(v, dict):
            if _has_rkey(v, '$import'):
                config[k] = parse_import(v)
            else:
                config[k] = _import(v)
        elif isinstance(v, list):
            for k2 in v:
                if isinstance(k2, dict):
                    _import(k2)
    return config


def _transform(config, dirname=None, run_generator=False, eval_python=False, run_compound=False):
    """Single-pass transformation of config tree."""
    if isinstance(config, dict):
        items = config.items()
    elif isinstance(config, list):
        items = enumerate(config)
    else:
        return config

    for k, v in items:
        tv = parse_param(v, dirname, run_generator, eval_python, run_compound)
        config[k] = _transform(tv, dirname, run_generator, eval_python, run_compound)

    return config


def has_key_deep(d, key):
    """Nested check for key in dict/list."""
    if isinstance(d, dict):
        items = d.items()
    elif isinstance(d, list):
        items = enumerate(d)
    else:
        return False
    for k, v in items:
        if isinstance(v, (dict, list)):
            if has_key_deep(v, key):
                return True
        if k == key:
            return True
    return False


def dict_merge(dct, merge_dct):
    """Recursive dict merge."""
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.abc.Mapping)):
            if '$sample' in dct[k]:
                dct[k] = merge_dct[k]
            else:
                dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def _rkey(d, key):
    """Get key name with backward compatibility (with or without $ prefix)."""
    if key in d:
        return key
    elif key[1:] in d:
        logger.warning('Deprecated replacement keyword. Prefix keyword with $.')
        return key[1:]
    else:
        logger.error('Replacement keyword not found.')
        return None


def _has_rkey(d, key):
    """Check for key with backward compatibility."""
    return key in d or key[1:] in d


def _has_rkey_deep(d, key):
    """Deep check for key with backward compatibility."""
    return has_key_deep(d, key) or has_key_deep(d, key[1:])
