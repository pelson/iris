# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.plot.pcolormesh` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
from iris.tests.unit.plot import TestGraphicStringCoord


if tests.MPL_AVAILABLE:
    import iris.plot as iplt


@tests.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    def test_pcolormesh_yaxis_labels(self):
        iplt.pcolormesh(self.cube, coords=('bar', 'str_coord'))
        actual = self.tick_loc_and_label('yaxis')
        expected = [(-0.5, 'a'), (0.0, 'a'), (0.5, 'b'), (1.0, 'b'),
                    (1.5, 'c'), (2.0, 'c'), (2.5, 'd'), (3.0, 'd'),
                    (3.5, '')]
        self.assertEqual(expected, actual)

    def test_pcolormesh_xaxis_labels(self):
        iplt.pcolormesh(self.cube, coords=('str_coord', 'bar'))
        actual = self.tick_loc_and_label('xaxis')
        expected = [(-0.5, 'a'), (0.0, 'a'), (0.5, 'b'), (1.0, 'b'),
                    (1.5, 'c'), (2.0, 'c'), (2.5, 'd'), (3.0, 'd'),
                    (3.5, '')]
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    tests.main()
