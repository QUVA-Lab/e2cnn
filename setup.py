
from setuptools import setup, find_packages

about = {}
with open("e2cnn/__about__.py") as fp:
    exec(fp.read(), about)

install_requires = [
    'torch',
    'numpy',
    'scipy',
]


setup_requires = []
tests_require = ['scikit-learn', 'scikit-image']

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__summary__'],
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__url__'],
    license=about['__license__'],
    long_description=long_description,
    packages=find_packages(),
    python_requires='>3.7',
    keywords=[
        'pytorch',
        'cnn',
        'convolutional-networks'
        'equivariant',
        'isometries',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
)
