from setuptools import setup, find_packages

setup(
    # Application name:
    name="pypls",

    # Version number (initial):
    version="0.1.0",

    # Packages
    packages=["pls"],

    # Data
    # package_data = {'pyneurovault':['template/*.html']},

    license="MIT",
    description="Pure python implementstion of Partial Least Squares inference",

    install_requires = ["pyprind>=2.9", "psutil>=2.2", "numpy>=1.9", "scipy", "scikit-learn>=0.15"],
    zip_safe=False
)