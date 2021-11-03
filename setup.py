# coding: utf-8

# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)

# Copyright (C) 2021 Jean-Philippe Meuret

# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

import pathlib as pl
import re
from setuptools import setup

# The directory containing this file
here = pl.Path(__file__).parent

# Retrieve version from __init__.py
with open(here / 'pyaudisam' / '__init__.py') as file:
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', file.read()).group(1)

# Retrieve install_requires from requirements.txt
with open(here / 'README.md') as file:
    long_desc = file.read()

# Retrieve install_requires from requirements.txt
with open(here / 'requirements.txt') as file:
    requirements = file.read().splitlines()

# This call to setup() does all the work
setup(name='pyaudisam', version=version, url='https://github.com/denmedius/pyaudisam',
      description='Distance Sampling automation through Distance sofware',
      long_description=long_desc, long_description_content_type='text/markdown',
      author='Jean-Philippe Meuret', author_email='jpmeuret@e.email',
      license='GPLv3+',
      classifiers=['Topic :: Software Development :: Libraries',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   'Intended Audience :: Science/Research',
                   'Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                   'Programming Language :: Python :: 3 :: Only',
                   'Programming Language :: Python :: 3.8',
                   'Environment :: Win32 (MS Windows)'],
      packages=['pyaudisam'],
      include_package_data=True,
      python_requires='>=3.8',
      install_requires=requirements,
      entry_points={'console_scripts': []})