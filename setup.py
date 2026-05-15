from setuptools import find_packages, setup

setup(
    name="inroom-air-filter",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=["essential_workers", "countries"],
)
