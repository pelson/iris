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
import iris.tests as tests
import iris.experimental.raster

import numpy as np
import PIL.Image


@iris.tests.skip_data
class TestGeoTiffExport(tests.GraphicsTest):
    def check_tiff_header(self, geotiff_fh, reference_filename):
        """
        Checks the given tiff file handle's metadata matches the 
        reference file contents.

        """
        im = PIL.Image.open(geotiff_fh)
        tiff_header = '\n'.join(str((tag, val)) for tag, val in sorted(im.tag.items()))
        reference_path = tests.get_result_path(reference_filename)
        
        self._check_same(tiff_header, reference_path, reference_filename, type_comparison_name='Tiff header')

    def test_load(self):
        cube = iris.load_cube(tests.get_data_path(('PP', 'hybrid_pressure',
                                                   'aklqaa.pyf1c10.pp')),
                              'surface_air_pressure')

        # Ensure longitude values are continuous and monotonically increasing,
        # and discard the 'half cells' at the top and bottom of the UM output
        # by extracting a subset
        east = iris.Constraint(longitude=lambda cell: cell < 180)
        non_edge = iris.Constraint(latitude=lambda cell: -90 < cell < 90)
        cube = cube.extract(east & non_edge)
        cube.coord('longitude').guess_bounds()
        cube.coord('latitude').guess_bounds()

        with self.temp_filename('.tif') as temp_filename:
            iris.experimental.raster.export_geotiff(cube, temp_filename)
            
            # Check the metadata is correct.
            with open(temp_filename) as fh:
                self.check_tiff_header(fh, ('experimental', 'raster', 
                                            'aklqaa.pyf1c10.subset.tif_header.txt'))

            # Check the data is correct.
            im = PIL.Image.open(temp_filename)
            im_data = np.array(im)
            
            np.testing.assert_array_equal(im_data, cube.data)


if __name__ == "__main__":
    tests.main()
