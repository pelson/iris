# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for :func:`iris.fileformats.pp_rules.convert`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import numpy as np

from iris.fileformats.pp_rules import convert
from iris.util import guess_coord_axis


class TestLBVC(tests.IrisTest):
    @staticmethod
    def _is_potm_level_coord(coord):
        return coord.standard_name == 'air_potential_temperature'

    @staticmethod
    def _is_model_level_number_coord(coord):
        return (coord.standard_name == 'model_level_number' and
                coord.units.is_dimensionless())

    @staticmethod
    def _is_reference_pressure_coord(coord):
        return (coord.name() == 'reference_pressure' and
                coord.units == 'Pa')

    @staticmethod
    def _is_sigma_coord(coord):
        return (coord.name() == 'sigma' and
                coord.units.is_dimensionless())

    def _test_for_coord(self, field, coord_predicate, expected_points,
                        expected_bounds):
        (factories, references, standard_name, long_name, units,
         attributes, cell_methods, dim_coords_and_dims,
         aux_coords_and_dims) = convert(field)

        # Check for one and only one matching coordinate.    
        coords_and_dims = dim_coords_and_dims + aux_coords_and_dims
        matching_coords = [coord for coord, _ in coords_and_dims if
                           coord_predicate(coord)]
        self.assertEqual(len(matching_coords), 1)
        coord = matching_coords[0]

        # Vertical coord should be Z-like, but this depends on consistent
        # setting of positive attribute.
        #self.assertEqual(guess_coord_axis(coord), 'Z')
        
        # Check points and bounds.
        if expected_points is not None:
            self.assertArrayEqual(coord.points, expected_points)

        if expected_bounds is None:
            self.assertIsNone(coord.bounds)
        else:
            self.assertArrayEqual(coord.bounds, expected_bounds)

    def test_soil_levels(self):
        level = 1234
        field = mock.MagicMock(lbvc=6, blev=level)
        self._test_for_coord(field, TestLBVC._is_model_level_number_coord,
                             expected_points=np.array([level]),
                             expected_bounds=None)

    def test_hybrid_pressure_levels(self):
        level = 5678
        field = mock.MagicMock(lbvc=9, lblev=level,
                               blev=20, brlev=23, bhlev=42,
                               bhrlev=45, brsvd=[17, 40])
        self._test_for_coord(field, TestLBVC._is_model_level_number_coord,
                             expected_points=np.array([level]),
                             expected_bounds=None)

    def test_hybrid_pressure_reference_pressure(self):
        field = mock.MagicMock(lbvc=9, lblev=5678,
                               blev=20, brlev=23, bhlev=12,
                               bhrlev=11, brsvd=[17, 13])
        self._test_for_coord(field, TestLBVC._is_reference_pressure_coord,
                             expected_points=np.array([1.0]),
                             expected_bounds=None)

    def test_hybrid_pressure_sigma(self):
        sigma_point = 12.0
        sigma_lower_bound = 11.0
        sigma_upper_bound = 13.0
        field = mock.MagicMock(lbvc=9, lblev=5678,
                               blev=20, brlev=23, bhlev=sigma_point,
                               bhrlev=sigma_lower_bound,
                               brsvd=[17, sigma_upper_bound])
        self._test_for_coord(field, TestLBVC._is_sigma_coord,
                             expected_points=np.array([sigma_point]),
                             expected_bounds=np.array([[sigma_lower_bound,
                                                        sigma_upper_bound]]))

    def test_potential_temperature_levels(self):
        potm_value = 27.32
        field = mock.MagicMock(lbvc=19, blev=potm_value)
        self._test_for_coord(field, TestLBVC._is_potm_level_coord,
                             expected_points=np.array([potm_value]),
                             expected_bounds=None)


if __name__ == "__main__":
    tests.main()
