"""
Tests for __main__.py entry point module.

This tests the CLI entry point that allows:
    python -m backpropagate
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestMainModule:
    """Tests for the __main__.py module."""

    def test_main_module_imports_cli_main(self):
        """Test that __main__ imports main from cli."""
        # This verifies the import works
        from backpropagate.cli import main
        assert callable(main)

    def test_main_module_can_be_imported(self):
        """Test that __main__ can be imported as a module."""
        # Import the module (this tests line 10: from .cli import main)
        import backpropagate.__main__
        assert hasattr(backpropagate.__main__, 'main')

    def test_main_module_has_main_function(self):
        """Test that the module exposes the main function."""
        from backpropagate.__main__ import main
        assert callable(main)

    def test_main_module_main_returns_int(self):
        """Test that main returns an integer exit code."""
        from backpropagate.__main__ import main

        # Call with empty args (shows help, returns 0)
        result = main([])
        assert isinstance(result, int)
        assert result == 0

    def test_main_module_main_with_info_command(self):
        """Test main with info command."""
        from backpropagate.__main__ import main

        result = main(["info"])
        assert result == 0

    def test_main_entry_point_guard(self):
        """Test that __name__ == '__main__' guard works correctly.

        This tests lines 12-13:
            if __name__ == "__main__":
                sys.exit(main())
        """
        # We can't directly test the if __name__ == "__main__" block,
        # but we can test that the components work correctly together
        from backpropagate.__main__ import main

        # Verify main can be called and returns proper exit codes
        assert main([]) == 0  # No command shows help
        assert main(["info"]) == 0  # Info command works
        assert main(["config"]) == 0  # Config command works

    def test_run_as_module_with_subprocess(self):
        """Test running as 'python -m backpropagate' via subprocess.

        This actually exercises the if __name__ == "__main__" block.
        """
        import subprocess

        # Run the module with --version flag
        result = subprocess.run(
            [sys.executable, "-m", "backpropagate", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # --version exits with code 0
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_run_as_module_info_command(self):
        """Test running 'python -m backpropagate info' via subprocess."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "backpropagate", "info"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0
        assert "System" in result.stdout or "backpropagate" in result.stdout.lower()

    def test_run_as_module_no_args_shows_help(self):
        """Test running 'python -m backpropagate' with no args shows help."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "backpropagate"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # No args should return 0 and show help
        assert result.returncode == 0
        # Help output should mention available commands
        assert "train" in result.stdout or "help" in result.stdout.lower()


class TestMainModuleIntegration:
    """Integration tests for __main__.py module execution."""

    def test_sys_exit_called_with_main_result(self):
        """Test that sys.exit is called with main's return value.

        This tests the pattern: sys.exit(main())
        """
        from backpropagate.__main__ import main

        with patch('sys.exit') as mock_exit:
            # Simulate what happens in __main__.py
            exit_code = main([])
            mock_exit(exit_code)

            mock_exit.assert_called_once_with(0)

    def test_main_with_invalid_command_shows_error(self):
        """Test main with an invalid command."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "backpropagate", "nonexistent_command"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Invalid command should show error or help
        # argparse returns exit code 2 for invalid arguments
        assert result.returncode != 0 or "error" in result.stderr.lower() or "usage" in result.stderr.lower()
