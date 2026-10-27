# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - David Yan, Jia Deng: primary authors

"""The optimized `PineconeFactory`, a `MonocotGrowthFactory` subclass, is built only
when MONOCOT is selected."""

import unittest

from infinigen._perf_selection import selected

PERF = selected("MONOCOT")


class PineconeSelection(unittest.TestCase):
    def test_factory_class_follows_the_selection(self):
        from infinigen.assets.objects.monocot.pinecone import PineconeFactory

        built = type(PineconeFactory(0))
        if PERF:
            self.assertTrue(built.__module__.startswith("infinigen.perf."), built)
            self.assertTrue(issubclass(built, PineconeFactory))
        else:
            self.assertIs(built, PineconeFactory)

    def test_native_kernels_only_referenced_by_the_optimized_copy(self):
        """The default monocot package has no compiled `_pinecone` kernel."""
        import infinigen.assets.objects.monocot as monocot

        self.assertFalse(
            (__import__("pathlib").Path(monocot.__path__[0]) / "_pinecone").exists(),
            "_pinecone belongs to the optimized copy only",
        )


if __name__ == "__main__":
    unittest.main()
