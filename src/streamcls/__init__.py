"""Shared library for the streaming classification pipeline.

Nothing heavy is imported here on purpose. The Django image installs
``streamcls[web]`` and must not pull a ~300 MB pyspark wheel, so ``pyspark`` is
confined to :mod:`streamcls.spark` and imported only by the services that need
a JVM.
"""

__version__ = "0.1.0"
