from setuptools import setup, find_packages
import re

with open("README.md", "r") as fh:
    long_description = fh.read()

def get_version():
    with open("fcmaes/__init__.py", "r") as f:
        match = re.search(r"(?m)^__version__\s*=\s*['\"](.+)['\"]$", f.read())
        return match.group(1)

setup(
    name='fcmaes',
    version=get_version(),
    description=('A Python 3 gradient-free optimization library.'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Dietmar Wolz',
    author_email='drdietmarwolz@yahoo.de',
    url='https://github.com/dietmarwo/fast-cma-es',
    license='MIT',
    packages=find_packages(),
    install_requires=[
          'numpy', 'scipy'
    ],
    classifiers = [
      "Intended Audience :: Manufacturing",
      "Intended Audience :: Financial and Insurance Industry",
      "Intended Audience :: Healthcare Industry",
      "Intended Audience :: Telecommunications Industry",
      "Intended Audience :: Information Technology",
      "Intended Audience :: Science/Research",
      "Intended Audience :: Education",
      "Topic :: Office/Business :: Financial",
      "Topic :: Office/Business :: Scheduling",
      "Topic :: Scientific/Engineering",
      "Topic :: Scientific/Engineering :: Information Analysis",
      "Topic :: Scientific/Engineering :: Mathematics",
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
      "Operating System :: OS Independent",
      "Programming Language :: Python :: 3",
      "Development Status :: 4 - Beta",
      "Environment :: Console",
      "License :: OSI Approved :: MIT License",
      ],
    keywords=["optimization", "multi-objective", "constraints", "CMA-ES", "BiteOpt", "MO-DE", "differential evolution", "annealing", "stochastic", "gradient free", "parallel execution", "boundary management"],
    include_package_data=True,
   )
