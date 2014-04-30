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
"""Unit tests for :class:`iris.fileformat.ff.ArakawaC`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np

from iris.fileformats.load_and_collate import collate, FieldGroup, Structure
from iris.fileformats.pp import PPField


class Test_FieldGroup(tests.IrisTest):
    def construct_field(self, **kwargs):
        kwargs.setdefault('lbexp', 0)
        time_elements = ['lbyr', 'lbmon', 'lbdat', 'lbhr', 'lbmin', 'lbsec',
                         'lbyrd', 'lbmond', 'lbdatd', 'lbhrd', 'lbmind', 'lbsecd',
                         'lbft']
        for elem in time_elements:
            kwargs.setdefault(elem, 0)
        kwargs.setdefault('lbuser', [0] * 5)
        kwargs.setdefault('lbrsvd', [0] * 4)
        kwargs.setdefault('blev', 0)
        return mock.Mock(spec=PPField, **kwargs)
    
    def test_simple_3d_structure(self):
        fields = []
        for pressure in [1000, 500, 250]:
            for day in [2, 3, 4, 5]:
                for experiment in [1, 2]:
                    fields.append(self.construct_field(blev=pressure, lbdat=day, lbexp=experiment))
        group = FieldGroup(None, fields)
        expected = {'lbexp': [[1, 2], [0, 1], True, True],
                    'blev': [[1000, 500, 250], [0, 8, 16], False, True],
                    'time': [[(0, 0, 2, 0, 0, 0),
                              (0, 0, 3, 0, 0, 0),
                              (0, 0, 4, 0, 0, 0),
                              (0, 0, 5, 0, 0, 0)],
                             [0, 2, 4, 6], True, True]}
        result = group.potential_structures()
        self.assertEqual(expected, result)

    def test_simple_2d_structure(self):
        fields = []
        for pressure in [1000, 500, 250]:
            for day in [2, 3, 4, 5]:
                fields.append(self.construct_field(blev=pressure, lbdat=day))
        group = FieldGroup(None, fields)
        expected = {'blev': [[1000, 500, 250], [0, 4, 8], False, True],
                    'time': [[(0, 0, 2, 0, 0, 0),
                              (0, 0, 3, 0, 0, 0),
                              (0, 0, 4, 0, 0, 0),
                              (0, 0, 5, 0, 0, 0)],
                             [0, 1, 2, 3], True, True]}
        result = group.potential_structures()
        self.assertEqual(expected, result)

    def test_simple_structure(self):
        fields = []
        for pressure in [1000, 500, 250]:
            fields.append(self.construct_field(blev=pressure))
        group = FieldGroup(None, fields)
        expected = {'blev': [[1000, 500, 250], [0, 1, 2], False, True]}
        self.assertEqual(expected, group.potential_structures())

    def test_shared_1d_structure(self):
        fields = []
        for pressure, day in zip([1000, 500, 250], [2, 3, 4, 5]):
            fields.append(self.construct_field(blev=pressure, lbdat=day))
        group = FieldGroup(None, fields)
        expected = {'blev': [[1000, 500, 250], [0, 1, 2], False, True],
                    'time': [[(0, 0, 2, 0, 0, 0),
                              (0, 0, 3, 0, 0, 0),
                              (0, 0, 4, 0, 0, 0)],
                             [0, 1, 2], False, True]}
        result = group.potential_structures()
        self.assertEqual(expected, result)
    
    def test_shared_2d_structure(self):
        fields = []
        for pressure, day in zip([1000, 500, 250], [2, 3, 4]):
            for experiment in [1, 2]:
                fields.append(self.construct_field(blev=pressure, lbdat=day, lbexp=experiment))
        group = FieldGroup(None, fields)
        expected = {'lbexp': [[1, 2], [0, 1], True, True],
                    'blev': [[1000, 500, 250], [0, 2, 4], False, True],
                    'time': [[(0, 0, 2, 0, 0, 0),
                              (0, 0, 3, 0, 0, 0),
                              (0, 0, 4, 0, 0, 0)],
                             [0, 2, 4], False, True]}
        result = group.potential_structures()
        self.assertEqual(expected, result)
    
    def test_2d_structure_with_non_viable(self):
        fields = []
        for experiment in [1, 2]:
            for pressure, day in zip([1000, 500, 250], [2, 3, 4]):
                fields.append(self.construct_field(blev=pressure, lbdat=day, lbexp=experiment))
        fields[2].lbdat = 6
        group = FieldGroup(None, fields)
        expected = {'lbexp': [[1, 2], [0, 3], False, True],
                     'blev': [[1000, 500, 250], [0, 1, 2], True, True],
                     'time': [[(0, 0, 2, 0, 0, 0),
                               (0, 0, 3, 0, 0, 0),
                               (0, 0, 6, 0, 0, 0)],
                              [0, 1, 2], True, False]}
        result = group.potential_structures()
        self.assertEqual(expected, result)


class Test_FieldGroup_structure(tests.IrisTest):
    def construct_FieldGroup(self, potential_structures):
        potential_structures = mock.Mock(return_value=potential_structures)
        group = mock.Mock(spec=FieldGroup,
                          potential_structures=potential_structures)
        group.structure = lambda *args, **kwargs: FieldGroup.structure(group,
                                                                       *args,
                                                                       **kwargs)
        return group
    
    def assertStructuresEqual(self, expected, result):
        self.assertEqual(len(expected), len(result))
        for structure1, structure2 in zip(expected, result):
            self.assertStructureEqual(structure1, structure2)
    
    def assertStructureEqual(self, struct1, struct2):
        assert isinstance(struct1, Structure)
        assert isinstance(struct2, Structure)
        for elem1, elem2 in zip(struct1, struct2):
            self.assertArrayEqual(elem1, elem2)
    
    def test_shared_dimension(self):
        potentials = {'blev': [[1000, 500, 250], [0, 1, 2], False, True],
                      'time': [[2, 3, 4], [0, 1, 2], False, True]}
        group = self.construct_FieldGroup(potentials)
        result = group.structure()
        expected = [Structure(elements='blev', sequence=[1000, 500, 250], stride=1),
                    Structure(elements='time', sequence=[2, 3, 4], stride=1)]
        self.assertStructuresEqual(expected, result)
    
    def test_simple_2d(self):
        # XXX: update time values
        potentials = {'blev': [[1000, 500, 250], [0, 4, 8], False, True],
                      'time': [[2, 3, 4, 5], [0, 1, 2, 3], True, True]}
        group = self.construct_FieldGroup(potentials)
        result = group.structure()
        expected = [Structure(elements='blev', sequence=[1000, 500, 250], stride=4),
                    Structure(elements='time', sequence=[2, 3, 4, 5], stride=1)]
        self.assertStructuresEqual(expected, result)


if __name__ == "__main__":
    tests.main()
