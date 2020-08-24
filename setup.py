"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call
import shlex
import sys
from warnings import warn


py_version = sys.version_info
if py_version.major < 3:
    raise RuntimeError("phygnn is only compatible with python 3!")

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "phygnn", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")

with open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    readme = f.read()

with open(os.path.join(here, "requirements.txt")) as f:
    install_requires = f.readlines()


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
    name="NREL-phygnn",
    version=version,
    description="Physics-Guided Neural Networks (phygnn)",
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.com/NREL/phygnn",
    packages=find_packages(),
    package_dir={"phygnn": "phygnn"},
    include_package_data=True,
    license="BSD 3-Clause",
    zip_safe=False,
    keywords="phygnn",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=install_requires,
    extras_require={
        "dev": ["flake8", "pre-commit", "pylint"],
    },
    cmdclass={"develop": PostDevelopCommand},
)
