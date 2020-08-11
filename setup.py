"""
setup.py
"""
import os
from codecs import open
from setuptools import setup, find_packages
import sys

py_version = sys.version_info
if py_version.major < 3:
    raise RuntimeError("phygnn is only compatible with python 3!")

try:
    from pypandoc import convert_text
except ImportError:
    convert_text = lambda string, *args, **kwargs: string

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "phygnn", "version.py"), encoding="utf-8") as f:
    version = f.read()

version = version.split('=')[-1].strip().strip('"').strip("'")

with open("README.rst", encoding="utf-8") as readme_file:
    readme = convert_text(readme_file.read(), "md", format="md")

setup(
    name="phygnn",
    version=version,
    description="Physics-Guided Neural Networks",
    long_description=readme,
    author="Grant Buster",
    author_email="grant.buster@nrel.gov",
    url="https://github.com/NREL/phygnn",
    packages=find_packages(),
    package_dir={"phygnn": "phygnn"},
    include_package_data=True,
    license="BSD license",
    zip_safe=False,
    keywords="phygnn",
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Modelers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    install_requires=["numpy>=1.16",
                      "pandas>=0.25",
                      "scipy>=1.3",
                      "matplotlib>=3.1",
                      "pytest>=5.2",
                      "scikit-learn>=0.22",
                      "tensorflow",
                      "ipython",
                      "notebook",
                      "psutil",
                      "pre-commit",
                      "flake8",
                      "pylint",
                      "NREL-rex",
                      ],
)
