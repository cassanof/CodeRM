from setuptools import setup, find_packages

reqs = open('requirements.txt').read().splitlines()

setup(
    name='codeprm',
    version='0.1.0',
    author='Federico Cassano',
    author_email='federico.cassano@federico.codes',
    packages=find_packages(),
    license='LICENSE',
    long_description=".",
    install_requires=reqs,
    python_requires='>=3.8',
)
