"""Archives store solutions found by a QD algorithm.

.. note:: After construction, each archive must be initialized by calling
    its ``initialize()`` method before it can be used. If using the optimizers
    in :mod:`ribs.optimizers`, this will be done automatically by the optimizer.

.. autosummary::
    :toctree:

    ribs.archives.GridArchive
    ribs.archives.CVTArchive
    ribs.archives.SlidingBoundariesArchive
    ribs.archives.ArchiveBase
    ribs.archives.AddStatus
"""
from dqd.ribs.archives._add_status import AddStatus
from dqd.ribs.archives._archive_base import ArchiveBase
from dqd.ribs.archives._cvt_archive import CVTArchive
from dqd.ribs.archives._grid_archive import GridArchive
from dqd.ribs.archives._sliding_boundaries_archive import SlidingBoundariesArchive

__all__ = [
    "GridArchive",
    "CVTArchive",
    "SlidingBoundariesArchive",
    "ArchiveBase",
    "AddStatus",
]
