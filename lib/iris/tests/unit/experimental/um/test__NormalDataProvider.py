# (C) British Crown Copyright 2015, Met Office
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
"""
Unit tests for :class:`iris.experimental.um._NormalDataProvider`.

"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from io import BytesIO
import mock
import numpy as np
from numpy.testing import assert_array_equal

from iris.experimental.um import Field, _NormalDataProvider


class Test_data_to_write(tests.IrisTest):
    def field_and_provider(self, word_width, orig_lbpack, new_lbpack):
        nx, ny = 3, 4
        lbnrec = nx * ny * 8
        if orig_lbpack == 2:
            dtype = '>f4'
        else:
            dtype = '>f8'

        # We make more bytes than are needed.
        source = BytesIO(np.arange(nx * ny, dtype=dtype).data)
        provider = _NormalDataProvider(source, 0, word_width, orig_lbpack)
        field = mock.Mock(lbpack=new_lbpack, lbnrec=lbnrec,
                          lbrow=ny, lbnpt=nx,
                          spec=Field)
        # Float datatype.
        field.lbuser1 = 1
        return field, provider

    def patch_np_fromfile(self):
        # Override numpy's faster fromfile with a call to fromstring after
        # reading the fh's bytes into memory.
        def slower_fromfile(fh, *args, **kwargs):
            return np.fromstring(fh.read(), *args, **kwargs)
        return mock.patch('numpy.fromfile', new=slower_fromfile)

    def test_load_pack_0_save_pack_1(self):
        field, provider = self.field_and_provider(8, 0, 1)
        with self.assertRaisesRegexp(ValueError, 'Cannot pack data'):
            provider.data_to_write(field)

    def test_load_pack_1_save_pack_1(self):
        # The writing of packed data should not involve an unpacking step
        # if the source and target are the same.
        field, provider = self.field_and_provider(8, 1, 1)
        expected = BytesIO(np.arange(12, dtype='>f8').data).read()
        actual = provider.data_to_write(field)
        self.assertEqual(actual, expected)

    def test_load_pack_2_save_pack_2(self):
        field, provider = self.field_and_provider(8, 2, 2)
        expected = np.arange(12, dtype='>f4')
        with self.patch_np_fromfile():
            assert_array_equal(provider.data_to_write(field), expected)

    def test_load_pack_2_save_pack_0(self):
        field, provider = self.field_and_provider(8, 2, 2)
        expected = np.arange(12, dtype='>f4')
        with self.patch_np_fromfile():
            actual = provider.data_to_write(field)
        assert_array_equal(actual, expected)
        # Although we would be writing to disk as 8bit, the data_to_write
        # method doesn't have that responsibility.
        self.assertEqual(actual.dtype, expected.dtype)


if __name__ == '__main__':
    tests.main()
