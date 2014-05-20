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
"""Unit tests for the :mod:`iris.plot` module."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.tests.stock import simple_2d
from iris.coords import AuxCoord


@tests.skip_plot
class TestGraphicStringCoord(tests.GraphicsTest):
    def setUp(self):
        self.cube = simple_2d(with_bounds=True)
        self.cube.add_aux_coord(AuxCoord(list('abcd'),
                                         long_name='str_coord'), 1)

    def tick_loc_and_label(self, axis_name):
        # Intentional lazy import so that subclasses can have an opportunity
        # to change the backend.
        import matplotlib.pyplot as plt
        plt.draw()
        axis = getattr(plt.gca(), axis_name)
        locations = axis.get_majorticklocs()
        labels = [tick.get_text() for tick in axis.get_ticklabels()]
        return zip(locations, labels)
