from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import find_packages, setup

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mise (efficient mesh extraction)
mise_module = Extension(
    "hsr.utils.libmise.mise",
    sources=["hsr/utils/libmise/mise.pyx"],
    include_dirs=[numpy_include_dir],
)

ext_modules = [mise_module]

setup(
    name="hsr",
    author="Lixin Xue, Chen Guo",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    cmdclass={"build_ext": build_ext},
    # setup_requires=["cython==0.29.20"],
    install_requires=[
        "simple-romp==1.1.3",
        "tensorboard==2.13.0",
        "numpy==1.23.1",
        "pytorch-lightning==2.0.5",
        # "opencv-python==4.7.0.72",
        "opencv-python-headless==4.7.0.72",
        "opencv-contrib-python-headless==4.7.0.72",
        # "opencv-python==4.5.5.64",
        "hydra-core==1.3.2",
        "scikit-image==0.21.0",
        "trimesh==3.21.7",
        "wandb==0.13.4",
        "matplotlib==3.5.3",
        "chumpy==0.70",
        "einops==0.6.0",
        "rtree==1.0.1",
        # "cython==0.29.20",
        "lpips==0.1.4",
        "pyliblzfse==0.4.1",
    ],
)
