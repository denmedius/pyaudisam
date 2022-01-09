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

# Submodule "utils": Everything useful that does not fit elsewhere ...

import types
import runpy
import pathlib as pl


def loadPythonData(path, **kwargs):

    """Load data from a python source file, as a types.SimpleNamespace for dot access to values by name

    Note: Special names are cleaned up from loaded data (starting with a '_' and usual module/variable names

    Parameters:
    :param path: Path to the python source file (if suffix omitted, .py is assumed)
    :param kwargs: Optional initial values for the loaded data

    :returns: tuple(explicit pl.Path of the module file, the types.SimpleNamespace of loaded data)"""

    # Determine python file name and absolute path
    path = pl.Path(path)
    if not path.suffix:
        path = path.with_suffix('.py')

    # Check file existence.
    if not path.is_file():
        return path, None

    # Load python source.
    usualModules = ['sys', 'os', 'pl', 'pathlib', 'dt', 'datetime', 'pd', 'pandas',
                    'math', 'np', 'numpy', 'rs', 'log', 'logger']
    data = {key: value for key, value in runpy.run_path(path.as_posix(), init_globals=kwargs).items()
            if not key.startswith('_') and key not in usualModules}

    return path, types.SimpleNamespace(**data)
