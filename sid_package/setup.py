from setuptools import setup, find_packages

setup(
    name='sid_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'networkx',
        'matplotlib',
    ],
    author='Hubery Hu',
    author_email='hubery.jiarui@gmail.com',
    description='A package for simulating influence and deinfluence in networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
