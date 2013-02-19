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
import os
import iris.tests as tests
import iris.experimental.raster


@iris.tests.skip_data
class TestGeoTiffExport(tests.GraphicsTest):
    def test_load(self):
        cube = iris.load_cube(tests.get_data_path(('PP', 'hybrid_pressure',
                                                   'aklqaa.pyf1c10.pp')),
                              'surface_air_pressure')

        fout = os.path.join("gdal", "aklqaa.pyf1c10.subset.tif")

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
            self.assertEqual(self.file_checksum(temp_filename),
                             self.file_checksum(tests.get_result_path(fout)))


if __name__ == "__main__":
    tests.main()
