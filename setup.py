from setuptools import setup, find_packages

setup(
  name = 'fttjax',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'Feature Tokenizer + Transformer - JAX',
  long_description_content_type = 'text/markdown',
  author = 'Tarun Agrawal',
  author_email = 'tarun@axra.org',
  url = 'https://github.com/github/fttransformer',
  keywords = [
    'artificial intelligence',
    'transformers',
    'attention mechanism',
    'tabular data'
  ],
  install_requires=[
    'einops>=0.5',
    'jax>=0.4',
    'flax>=0.7',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.11',
  ],
)