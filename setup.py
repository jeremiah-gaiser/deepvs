from setuptools import setup, find_packages
import yaml

setup(
        name ='deepvs',
        version ='0.0.1',
        packages = find_packages(),
        entry_points = {
            'console_scripts': [
                'cli_test = code.pdbbind_data_processing.cli_test:main'
            ]
        },
)