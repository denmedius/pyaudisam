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
import pandas as pd


def loadPythonData(path, **kwargs):

    """Load data from a python source file, as a types.SimpleNamespace for dot access to values by name

    Note: Special names (starting with a '_' + usual module/variable names) are removed from loaded data
          before returning.

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


def mapDataFrame(df, func, na_action=None, **kwargs):
    """Wrapper to pandas DataFrame.applymap renamed to map from 2.1.0"""
    return df.map(func, na_action, **kwargs) if pd.__version__ >= '2.1.0' \
        else df.applymap(func, na_action, **kwargs)


KPandasFreqAliases = {} if pd.__version__ < '2.2.0' \
                     else {'H': 'h', 'T': 'min', 'S': 's', 'L': 'ms', 'U': 'us', 'N': 'ns'}

def pandasFreqAlias(freq):
    """Pandas freq alias wrapper managing deprecations"""
    return KPandasFreqAliases.get(freq, freq)
