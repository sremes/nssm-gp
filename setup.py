from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='nssm-gp',
      version='0.1',
      description='Non-stationary spectral kernels for GPflow',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/sremes/nssm-gp',
      author='Sami Remes',
      author_email='sami.remes@gmail.com',
      license='MIT',
      packages=['nssm_gp'],
      install_requires=[
          'gpflow',
          'pandas',
          'scikit-learn',
          'pytables',
          'h5py',
          'numpy', 
          'tensorflow',
      ],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Artificial Intelligence"
      ],
)
