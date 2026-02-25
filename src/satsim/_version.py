try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("satsim")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
except ImportError:
    __version__ = "0.0.0-dev"
