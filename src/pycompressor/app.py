# implement ValidPhys application

import re
import sys
import shutil
import pathlib
import logging
import hashlib
import warnings

from validphys.app import App
from reportengine import colors
from reportengine.compat import yaml
from validphys.config import Environment, Config
from validphys.config import EnvironmentError_, ConfigError

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

APP_CONFIG = dict(actions_=["compressing"])
APP_PROVIDERS = ["pycompressor.compressing"]
INPUT_FOLDER = "input"


class CompressorError(Exception):
    """Compressor error."""

    pass


class CompressorEnvironment(Environment):
    """Container for information to be filled at run time"""

    def init_output(self):
        # check file exists, is a file, has extension.
        if not self.config_yml.exists():
            raise CompressorError("Invalid runcard. File not found.")
        else:
            if not self.config_yml.is_file():
                raise CompressorError("Invalid runcard. Must be a file.")
        # Check if results folder exists
        self.output_path = pathlib.Path(self.output_path).absolute()
        try:
            self.output_path.mkdir(exist_ok=True)
        except OSError as err:
            raise EnvironmentError_(err) from err
        # Create input folder
        self.input_folder = self.output_path / INPUT_FOLDER
        self.input_folder.mkdir(exist_ok=True)


class CompressorConfig(Config):
    """Specialization for yaml parsing"""

    @classmethod
    def from_yaml(cls, o, *args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", yaml.error.MantissaNoDotYAML1_1Warning)
                file_content = yaml.safe_load(o, version="1.1")
        except yaml.error.YAMLError as e:
            raise ConfigError(f"Failed to parse yaml file: {e}")
        if not isinstance(file_content, dict):
            raise ConfigError(
                f"Expecting input runcard to be a mapping, "
                f"not '{type(file_content)}'."
            )
        file_content.update(APP_CONFIG)
        return cls(file_content, *args, **kwargs)


class CompressorApp(App):
    """The class which parsers and performs the fit"""

    environment_class = CompressorEnvironment
    config_class = CompressorConfig

    def __init__(self):
        super(CompressorApp, self).__init__(name="compressor", providers=APP_PROVIDERS)

    @property
    def argparser(self):
        parser = super().argparser
        parser.add_argument("-o", "--output", help="Output folder", default=None)
        return parser

    def get_commandline_arguments(self, cmdline=None):
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem
        return args

    def run(self):
        try:
            self.environment.config_yml = pathlib.Path(
                self.args["config_yml"]
            ).absolute()
            super().run()
        except CompressorError as err:
            log.error(f"Error in pyCompressor:\n{err}")
            sys.exit(1)
        except Exception as err:
            log.critical("Bug in pyCompressor ocurred. Please report it.")
            print(
                colors.color_exception(err.__class__, err, err.__traceback__),
                file=sys.stderr,
            )
            sys.exit(1)


def main():
    application = CompressorApp()
    application.main()