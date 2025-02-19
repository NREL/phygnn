"""
setup.py
"""
import shlex
from subprocess import check_call
from warnings import warn

from setuptools import setup
from setuptools.command.develop import develop


class PostDevelopCommand(develop):
    """
    Class to run post setup commands
    """

    def run(self):
        """
        Run method that tries to install pre-commit hooks
        """
        try:
            check_call(shlex.split("pre-commit install"))
        except Exception as e:
            warn("Unable to run 'pre-commit install': {}"
                 .format(e))

        develop.run(self)


setup(
    test_suite="tests",
    cmdclass={"develop": PostDevelopCommand},
)
