#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ["ijson", "numpy", "pandas", "tqdm", "scikit-learn", "matplotlib"]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Kayané Elmayan Robach",
    author_email='kaya.robach@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Learning Experience Analysis Toolbox contains tools to analyse learning experience data using the pandas library. This package is developped for EvidenceB leraning experience data.",
    entry_points={
        'console_scripts': [
            'LrnXPAnaToolbox=LrnXPAnaToolbox.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='kayatoolbox',
    name='kayatoolbox',
    packages=find_packages(include=['kayatoolbox', 'kayatoolbox.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/robachkaya/LrnXPAnaToolbox',
    version='0.4.1',
    zip_safe=False,
)
