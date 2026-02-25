"""Tests for CLI."""

import pytest
from click.testing import CliRunner

from satsim.cli import main


class TestCLI:
    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ['version'])
        assert result.exit_code == 0

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'SatSim' in result.output
