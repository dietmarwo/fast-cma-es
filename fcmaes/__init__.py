# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

__version__ = '2.0.0'

try:
    from . import _native as _fcmaes_native
except ModuleNotFoundError:
    _fcmaes_native = None

__all__ = [
    'cmaes',
    'cmaescpp',
    'crmfnes',
    'crfmnescpp',
    'de',
    'dacpp',
    'decpp',
    'diversifier',
    'bitecpp',
    'csmacpp',
    'retry',
    'advretry',
    'mapelites',
    'multiretry',
    'mode',
    'modecpp',
    'moretry',
    'pygmoretry',
    'optimizer',
    'astro',
    'evaluator',
    'testfun',
    'journal',
]

del _fcmaes_native
