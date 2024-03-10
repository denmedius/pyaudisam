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

# Package main script, for when pyaudisam is invoked through "python -m"

# Usage: python -m pyaudisam --help

import sys

from .main import main

rc = main(sys.argv[1:])  # In case of an uncaught exception, will sys.exit(1)

sys.exit(rc)
