"""Main CLI command"""

import abc
import argparse
from importlib import resources
import pathlib
import shutil
import sys
from typing import Final, cast

import hydra
from tensorial import reaxkit as rkit

from . import config

COMMAND: Final[str] = "command"
TRAIN: Final[str] = "train"
PREDICT: Final[str] = "predict"
TRAIN_SCRIPT_DEFAULT: Final[str] = "examples/train.yaml"
EVAL_SCRIPT_DEFAULT: Final[str] = "examples/eval.yaml"


class Command(abc.ABC):
    NAME: Final[str]

    @abc.abstractmethod
    def register(self, commands):
        """Register the command with argparse"""

    @abc.abstractmethod
    def run(self, args, rest, /):
        """Run this command"""


class Train(Command):
    NAME = "train"

    def register(self, commands):
        # The 'train' command
        train_parser = commands.add_parser(self.NAME, help="Train a model")
        train_parser.add_argument(
            "-i",
            "--input",
            nargs="?",
            type=pathlib.Path,
            help="Input file with training details",
            default=TRAIN_SCRIPT_DEFAULT,
        )

    def run(self, args, _rest, /):
        # Set the command line arguments to what remains so hydra can deal with it
        sys.argv = sys.argv[0:1] + _rest

        script_path: pathlib.Path = args.input
        hydra_fn = hydra.main(
            version_base="1.3",
            config_path=str(script_path.parent.absolute()),
            config_name=script_path.stem,
        )(rkit.train.main)
        hydra_fn()


class Predict(Command):
    NAME = PREDICT

    def register(self, commands):
        # The 'predict' command
        train_parser = commands.add_parser(self.NAME, help="Make predictions using a trained model")
        train_parser.add_argument(
            "-i",
            "--input",
            nargs="?",
            type=pathlib.Path,
            help="Input file with evaluation details",
            default=EVAL_SCRIPT_DEFAULT,
        )

    def run(self, args, _rest, /):
        # Set the command line arguments to what remains so hydra can deal with it
        sys.argv = sys.argv[0:1] + _rest

        script_path = cast(pathlib.Path, args.input)
        if script_path.is_dir():
            script_path = script_path / config.DEFAULT_CONFIG_FILE
            if not script_path.is_file():
                print(f"Could not find configuration file: {script_path}")
                sys.exit(1)

        hydra_fn = hydra.main(
            version_base="1.3",
            config_path=str(script_path.parent.absolute()),
            config_name=script_path.stem,
        )(rkit.evaluate.main)
        hydra_fn()


class Examples(Command):
    NAME = "examples"

    def _list_examples(self):
        assets_dir = resources.files("e3response").joinpath("examples")
        with resources.as_file(assets_dir) as source_path:
            examples = [d.name for d in source_path.iterdir() if d.is_dir()]
            print("Available examples:")
            for example in examples:
                print(f" - {example}")

    def _init_example(self, example_name: str, dest: pathlib.Path):
        assets_dir = resources.files("e3response").joinpath("examples")
        with resources.as_file(assets_dir) as source_path:
            example_path = source_path / example_name
            if not example_path.exists():
                available = ", ".join([d.name for d in source_path.iterdir() if d.is_dir()])
                print(f"Example '{example_name}' not found. Available examples: {available}")
                sys.exit(1)

            dest_folder = dest / example_name
            print(f"Initializing example '{example_name}' in {dest_folder}")
            dest_folder.mkdir(parents=True, exist_ok=True)

            # Copy contents of example_path into dest_folder
            for item in example_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest_folder / item.name)
                elif item.is_dir():
                    shutil.copytree(item, dest_folder / item.name, dirs_exist_ok=True)

    def register(self, commands):
        # The 'examples' command
        parser = commands.add_parser(self.NAME, help="Manage examples")
        subparsers = parser.add_subparsers(dest="subcommand", help="Examples subcommands")

        # list command
        subparsers.add_parser("list", help="List all available examples")

        # init command
        init_parser = subparsers.add_parser("init", help="Initialize an example")
        init_parser.add_argument("name", help="Name of the example to initialize")
        init_parser.add_argument(
            "output_dir",
            nargs="?",
            default=".",
            type=pathlib.Path,
            help="Directory to create the example in (defaults to current directory)",
        )

    def run(self, args, rest, /):
        # 1. Handle explicitly parsed subcommands ('list' or 'init')
        if args.subcommand == "list":
            self._list_examples()

        elif args.subcommand == "init":
            self._init_example(args.name, args.output_dir)

        # 2. Handle missing subcommands (`e3response examples` or `e3response examples bto`)
        else:
            if rest:
                # If they typed `e3response examples bto`, 'bto' ends up in `rest`
                example_name = rest[0]
                dest = pathlib.Path(rest[1]) if len(rest) > 1 else pathlib.Path(".")

                # Verify the implicit argument is actually an example
                assets_dir = resources.files("e3response").joinpath("examples")
                with resources.as_file(assets_dir) as source_path:
                    if (source_path / example_name).is_dir():
                        self._init_example(example_name, dest)
                    else:
                        print(f"Unknown command or example: {example_name}\n")
                        self._list_examples()
                        sys.exit(1)
            else:
                # No subcommand and no extra arguments -> Default to 'list'
                self._list_examples()


COMMANDS = {
    Train.NAME: Train(),
    Predict.NAME: Predict(),
    Examples.NAME: Examples(),
}


def main_cli():
    parser = argparse.ArgumentParser("e3response")
    commands = parser.add_subparsers(dest=COMMAND, required=True)

    for command in COMMANDS.values():
        command.register(commands)

    # Parse the args
    args, _rest = parser.parse_known_args()

    try:
        return COMMANDS[args.command].run(args, _rest)
    except KeyError:
        raise ValueError(f"Unknown command: {args.command}") from None


if __name__ == "__main__":
    main_cli()
