from setuptools import setup, find_packages

setup(
    name="autogpt",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'autogpt=autogpt.__main__:main',
        ],
    },
    install_requires=[
        # Add your project dependencies here
    ],
    # Add other metadata as needed
)
