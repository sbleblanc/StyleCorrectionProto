from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

ext_modules = []
if use_cython:
    ext_modules += cythonize("stylecorrection/utils/cython_utils.pyx")
else:
    ext_modules += [Extension('stylecorrection.utils.cython_utils', ['stylecorrection/utils/cython_utils.c'])]

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="stylecorrectionproto",
    version="1.0.0",
    author="Samuel Beland-Leblanc",
    author_email="samuel.beland.leblanc@gmail.com",
    description="A style correction prototype using sequence to sequence Transformer neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sbleblanc/StyleCorrectionProto",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'torch>=1.4.0',
        'spacy>=2.2.3',
        'fastBPE>=0.1.0',
        'h5py>=2.10.0',
        'PyYAML>=5.3',
        'numpy>=1.18.1',
        'scipy>=1.4.1'
    ],
    ext_modules=ext_modules
)
