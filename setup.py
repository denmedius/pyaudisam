import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# This call to setup() does all the work
setup(
    name="pyaudisam",
    version="0.9.0",
    python_requires='>=3.8.0',
    description="Distance Sampling automation through Distance sofware",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/denmedius/pyaudisam",
    author="Jean-Philippe Meuret",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["pyaudisam"],
    include_package_data=True,
    install_requires=["pandas", "matplotlib", "jinja2", "zoopt"],
    entry_points={
        "console_scripts": []
    },
)