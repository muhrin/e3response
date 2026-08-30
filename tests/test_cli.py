from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from e3response.cli import Examples

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def mock_resources():
    with patch("e3response.cli.resources") as mock_res:
        # Mock directory structure for examples
        mock_examples_dir = MagicMock()
        mock_res.files.return_value.joinpath.return_value = mock_examples_dir

        # Mock paths
        mock_bto = MagicMock()
        mock_bto.name = "bto"
        mock_bto.is_dir.return_value = True

        mock_qm9 = MagicMock()
        mock_qm9.name = "qm9-nmr"
        mock_qm9.is_dir.return_value = True

        mock_examples_dir.iterdir.return_value = [mock_bto, mock_qm9]

        # Ensure as_file context manager works
        mock_res.as_file.return_value.__enter__.return_value = mock_examples_dir

        yield mock_res


def test_examples_list(capsys, mock_resources):
    cmd = Examples()
    # Simulate arguments that main_cli would produce for "examples list"
    args = Namespace(command="examples", subcommand="list")
    cmd.run(args, [])
    captured = capsys.readouterr()
    assert "Available examples:" in captured.out
    assert "- bto" in captured.out
    assert "- qm9-nmr" in captured.out


@patch("e3response.cli.shutil")
def test_examples_init(mock_shutil, tmp_path, mock_resources):
    cmd = Examples()

    # Mock example file structure for 'bto'
    mock_bto_dir = MagicMock()
    mock_bto_dir.exists.return_value = True

    # Mock content of example
    mock_file = MagicMock()
    mock_file.name = "test.yaml"
    mock_file.is_file.return_value = True
    mock_bto_dir.iterdir.return_value = [mock_file]

    # Setup mock_resources to return this mock_bto_dir when path is joined
    mock_examples_dir = mock_resources.files.return_value.joinpath.return_value
    mock_examples_dir.__truediv__.return_value = mock_bto_dir

    output_dir = tmp_path / "output"

    # Simulate arguments for "examples init bto output"
    args = Namespace(command="examples", subcommand="init", name="bto", output_dir=output_dir)
    # Run init
    cmd.run(args, [])

    # Verify shutil.copy2 was called
    mock_shutil.copy2.assert_called_once()
    assert mock_shutil.copy2.call_args[0][1].name == "test.yaml"
