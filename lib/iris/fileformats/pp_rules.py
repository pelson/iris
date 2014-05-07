# (C) British Crown Copyright 2013 - 2014, Met Office
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

# Historically this was auto-generated from
# SciTools/iris-code-generators:tools/gen_rules.py

import warnings

import numpy as np

from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.rules import Factory, Reference, ReferenceTarget
from iris.fileformats.um_cf_map import LBFC_TO_CF, STASH_TO_CF
from iris.unit import Unit
import iris.fileformats.pp
import iris.unit


def _model_level_number(field):
    """
    Return the model level number of a field.

    Args:

    * field (:class:`iris.fileformats.pp.PPField`)
        PP field to inspect.

    Returns:
        Model level number (integer).

    """
    # See Word no. 33 (LBLEV) in section 4 of UM Model Docs (F3).
    SURFACE_AND_ZEROTH_RHO_LEVEL_LBLEV = 9999

    if field.lblev == SURFACE_AND_ZEROTH_RHO_LEVEL_LBLEV:
        model_level_number = 0
    else:
        model_level_number = field.lblev

    return model_level_number


def scalar_time_coords(lbproc, lbtim, lbcode, hour_time_unit, t1, t2, lbft):
    # n.b. All inputs must be scalars.

    aux_coords_and_dims = []
    
    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 0) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        aux_coords_and_dims.append((DimCoord(hour_time_unit.date2num(t1), standard_name='time', units=hour_time_unit), None))

    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 1) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        hours_since_t2 = iris.unit.Unit('hours since %s' % t2, calendar=hour_time_unit.calendar)
        aux_coords_and_dims.append((DimCoord(hours_since_t2.date2num(t1), standard_name='forecast_period', units='hours'), None))
        aux_coords_and_dims.append((DimCoord(hour_time_unit.date2num(t1), standard_name='time', units=hour_time_unit), None))
        aux_coords_and_dims.append((DimCoord(hour_time_unit.date2num(t2), standard_name='forecast_reference_time', units=hour_time_unit), None))

    if \
            (lbtim.ib == 2) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        t_unit = hour_time_unit
        t1_hours = t_unit.date2num(t1)
        t2_hours = t_unit.date2num(t2)
        period = t2_hours - t1_hours
        aux_coords_and_dims.append((
            DimCoord(standard_name='forecast_period', units='hours',
                     points=lbft - 0.5 * period,
                     bounds=[lbft - period, lbft]),
            None))
        aux_coords_and_dims.append((
            DimCoord(standard_name='time', units=t_unit,
                     points=0.5 * (t1_hours + t2_hours),
                     bounds=[t1_hours, t2_hours]),
            None))
        aux_coords_and_dims.append((DimCoord(hour_time_unit.date2num(t2) - lbft, standard_name='forecast_reference_time', units=hour_time_unit), None))

    if \
            (lbtim.ib == 3) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        t_unit = hour_time_unit
        t1_hours = t_unit.date2num(t1)
        t2_hours = t_unit.date2num(t2)
        period = t2_hours - t1_hours
        aux_coords_and_dims.append((
            DimCoord(standard_name='forecast_period', units='hours',
                     points=lbft, bounds=[lbft - period, lbft]),
            None))
        aux_coords_and_dims.append((
            DimCoord(standard_name='time', units=t_unit,
                     points=t2_hours, bounds=[t1_hours, t2_hours]),
            None))
        aux_coords_and_dims.append((DimCoord(hour_time_unit.date2num(t2) - lbft, standard_name='forecast_reference_time', units=hour_time_unit), None))
    
    return aux_coords_and_dims



def scalar_vertical_coords():
    # n.b. All inputs must be scalars.

    aux_coords_and_dims = ()

    if \
            (f.lbvc == 1) and \
            (not (str(f.stash) in ['m01s03i236', 'm01s03i237', 'm01s03i245', 'm01s03i247', 'm01s03i250'])) and \
            (f.blev != -1):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='height', units='m', attributes={'positive': 'up'}), None))

    if str(f.stash) in ['m01s03i236', 'm01s03i237', 'm01s03i245', 'm01s03i247', 'm01s03i250']:
        aux_coords_and_dims.append((DimCoord(1.5, standard_name='height', units='m', attributes={'positive': 'up'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2):
        aux_coords_and_dims.append((DimCoord(_model_level_number(f), standard_name='model_level_number', attributes={'positive': 'down'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2) and \
            (f.brsvd[0] == f.brlev):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='depth', units='m', attributes={'positive': 'down'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2) and \
            (f.brsvd[0] != f.brlev):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='depth', units='m', bounds=[f.brsvd[0], f.brlev], attributes={'positive': 'down'}), None))

    # soil level
    if len(f.lbcode) != 5 and f.lbvc == 6:
        aux_coords_and_dims.append((DimCoord(_model_level_number(f), long_name='soil_model_level_number', attributes={'positive': 'down'}), None))

    if \
            (f.lbvc == 8) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and 1 not in [f.lbcode.ix, f.lbcode.iy])):
        aux_coords_and_dims.append((DimCoord(f.blev, long_name='pressure', units='hPa'), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 19):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='air_potential_temperature', units='K', attributes={'positive': 'up'}), None))

#    # Hybrid pressure coordinate
#    if f.lbvc == 9:
#        model_level_number = DimCoord(_model_level_number(f),
#                                      standard_name='model_level_number',
#                                      attributes={'positive': 'up'})
#        # The following match the hybrid height scheme, but data has the
#        # blev and bhlev values the other way around.
#        #level_pressure = DimCoord(f.blev,
#        #                          long_name='level_pressure',
#        #                          units='Pa',
#        #                          bounds=[f.brlev, f.brsvd[0]])
#        #sigma = AuxCoord(f.bhlev,
#        #                 long_name='sigma',
#        #                 bounds=[f.bhrlev, f.brsvd[1]])
#        level_pressure = DimCoord(f.bhlev,
#                                  long_name='level_pressure',
#                                  units='Pa',
#                                  bounds=[f.bhrlev, f.brsvd[1]])
#        sigma = AuxCoord(f.blev,
#                         long_name='sigma',
#                         bounds=[f.brlev, f.brsvd[0]])
#        aux_coords_and_dims.extend([(model_level_number, None),
#                                    (level_pressure, None),
#                                    (sigma, None)])
#        factories.append(Factory(HybridPressureFactory,
#                                 [{'long_name': 'level_pressure'},
#                                  {'long_name': 'sigma'},
#                                  Reference('surface_air_pressure')]))

#    if f.lbvc == 65:
#        aux_coords_and_dims.append((DimCoord(_model_level_number(f), standard_name='model_level_number', attributes={'positive': 'up'}), None))
#        aux_coords_and_dims.append((DimCoord(f.blev, long_name='level_height', units='m', bounds=[f.brlev, f.brsvd[0]], attributes={'positive': 'up'}), None))
#        aux_coords_and_dims.append((AuxCoord(f.bhlev, long_name='sigma', bounds=[f.bhrlev, f.brsvd[1]]), None))
#        factories.append(Factory(HybridHeightFactory, [{'long_name': 'level_height'}, {'long_name': 'sigma'}, Reference('orography')]))

    return aux_coords_and_dims


def scalar_realization_coords(lbrsvd_3):
    # Coordinates coming from the lbrsvd[3] element (ensemble)
    # n.b. All inputs must be scalars.
    if lbrsvd_3 != 0:
        return (DimCoord(lbrsvd_3, standard_name='realization'), None)
    return []


def scalar_pseudo_level_coords(lbuser_4):
    # Coordinates coming from the lbuser[4] element (pseudo_level)
    # n.b. All inputs must be scalars.
    if lbuser_4 != 0:
        return ((DimCoord(lbuser_4, long_name='pseudo_level', units='1'), None), )
    return []


def vector_realization_coords(lbrsvd_3_and_dims):
    # Coordinates coming from the lbrsvd[3] element (ensemble)
    # Returns an iterable of coordinates.
    lbrsvd_3, dims = lbrsvd_3_and_dims
    if np.any(lbrsvd_3 != 0):
        return ([AuxCoord(lbrsvd_3, standard_name='realization'), dims], )
    return []


def vector_time_coords(lbproc, lbtim, lbcode, hour_time_unit, t1_and_dims, t2_and_dims, lbft_and_dims):
    # Scalars: lbproc, lbtim, lbcode, hour_time_unit
    # Potential vectors: t1, t2, lbft
    
    # n.b. input dims are sorted on entry - the inputs cannot have different order mapped dimensions.
    
    t1, t1_dims = t1_and_dims
    t2, t2_dims = t2_and_dims
    lbft, lbft_dims = lbft_and_dims
    unique_dims = sorted(set(tuple(t1_dims) + tuple(t2_dims) + tuple(lbft_dims)))

    t1 = np.array(t1, ndmin=1)
    t2 = np.array(t2, ndmin=1) 
    lbft = np.array(lbft, ndmin=1)

    # Compute the full shape of the data spanned by the input vectors.
    full_shape = [None] * len(unique_dims)
    for name, array, dims in [('t1', t1, t1_dims),
                              ('t2', t2, t2_dims),
                              ('lbft', lbft, lbft_dims)]:
        if (len(dims) or 1) != len(array.shape) and not (array.size == 1 and len(dims) == 0):
            print len(array.shape), len(dims), dims, name
            raise ValueError('Dims and shape mismatch for coord {}.'.format(name))
        for dim, length in zip(dims, array.shape):
            if full_shape[unique_dims.index(dim)] is None:
                full_shape[unique_dims.index(dim)] = length
            else:
                if full_shape[unique_dims.index(dim)] != length:
                    raise ValueError('Incompatible shapes.')

    def combined_dims(dims_tuples):
        import itertools
        return sorted(set(itertools.chain.from_iterable(tuple(dims_tuples))))
    
    def filled_array(a, a_dims, dims_tuples):
        import numpy.lib.stride_tricks
        
        all_dims = combined_dims(tuple(dims_tuples) + (a_dims, ))
        all_dims = [unique_dims.index(dim) for dim in all_dims] 
        broadcast_shape = [1] * len(full_shape)
        for length, dim in zip(a.shape, a_dims):
            broadcast_shape[unique_dims.index(dim)] = length
        
        broadcast_shape = [shape for dim, shape in enumerate(broadcast_shape) if dim in all_dims]
        broadcast_a = a.reshape(broadcast_shape)

        # Now turn this into a potentially 0-strided view of the array.
        new_strides = [0] * len(all_dims)
        for dim, (length, stride) in enumerate(zip(broadcast_a.shape, broadcast_a.strides)):
            if length == 1 and full_shape[dim] != 1:
                new_strides[unique_dims.index(dim)] = 0
            else:
                new_strides[unique_dims.index(dim)] = stride
        return np.lib.stride_tricks.as_strided(broadcast_a, strides=new_strides,
                                                            shape=[full_shape[dim] for dim in all_dims])
    aux_coords_and_dims = []
    
    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 0) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        raise ValueError('Vectorize needed')
        aux_coords_and_dims.append(AuxCoord(hour_time_unit.date2num(t1), standard_name='time', units=hour_time_unit))

    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 1) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        # XXX Vectorize?
        assert len(t2) == 1
        hours_since_t2 = iris.unit.Unit('hours since %s' % t2[0], calendar=hour_time_unit.calendar)
        aux_coords_and_dims.append([AuxCoord(hours_since_t2.date2num(t1), standard_name='forecast_period', units='hours'), t1_dims])
        aux_coords_and_dims.append([AuxCoord(hour_time_unit.date2num(t1), standard_name='time', units=hour_time_unit), t1_dims])
        aux_coords_and_dims.append([AuxCoord(hour_time_unit.date2num(t2), standard_name='forecast_reference_time', units=hour_time_unit), t2_dims])

    if \
            (lbtim.ib == 2) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        t_unit = hour_time_unit

        t1_b = filled_array(t1, t1_dims, [t2_dims])
        t2_b = filled_array(t2, t2_dims, [t1_dims])

        t2_w_t1_lbft_b = filled_array(t2, t2_dims, [t1_dims, lbft_dims])
        t1_w_t2_lbft_b = filled_array(t1, t1_dims, [t2_dims, lbft_dims])

        t2_w_lbft_b = filled_array(t2, t2_dims, [lbft_dims])
        lbft_f = filled_array(lbft, lbft_dims, [t1_dims, t2_dims])
        lbft_w_t2_b = filled_array(lbft, lbft_dims, [t2_dims])
        
        t1_hours = np.array(t_unit.date2num(t1_b), ndmin=1)
        t2_hours = np.array(t_unit.date2num(t2_b), ndmin=1)
        
        
        period = t_unit.date2num(t2_w_t1_lbft_b) - t_unit.date2num(t1_w_t2_lbft_b)

        fp_points = lbft_f - 0.5 * period
        fp_bounds = np.concatenate([lbft_f - period, lbft_f]).reshape(fp_points.shape + (2,))
        time_points = np.array(0.5 * (t1_hours + t2_hours))
        time_bounds = np.concatenate([t1_hours, t2_hours]).reshape(time_points.shape + (2,))
        
        aux_coords_and_dims.append([AuxCoord(standard_name='forecast_period', units='hours',
                                             points=fp_points, bounds=fp_bounds),
                                    combined_dims([t1_dims, t2_dims, lbft_dims])])
        aux_coords_and_dims.append([AuxCoord(standard_name='time', units=t_unit,
                                             points=time_points, bounds=time_bounds),
                                    combined_dims([t1_dims, t2_dims])])
        aux_coords_and_dims.append([AuxCoord(hour_time_unit.date2num(t2_w_lbft_b) - lbft_w_t2_b,
                                             standard_name='forecast_reference_time',
                                             units=hour_time_unit), combined_dims([t2_dims, lbft_dims])])

    if \
            (lbtim.ib == 3) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        t_unit = hour_time_unit
        t1_hours = t_unit.date2num(t1)
        t2_hours = t_unit.date2num(t2)
        period = t2_hours - t1_hours
        raise ValueError('Vectorize needed')
        aux_coords_and_dims.append(AuxCoord(standard_name='forecast_period', units='hours', points=lbft, bounds=[lbft - period, lbft]))
        aux_coords_and_dims.append(AuxCoord(standard_name='time', units=t_unit, points=t2_hours, bounds=[t1_hours, t2_hours]))
        aux_coords_and_dims.append(AuxCoord(hour_time_unit.date2num(t2) - lbft, standard_name='forecast_reference_time', units=hour_time_unit))
    
    return aux_coords_and_dims



def convert(f, exclude=None):
    factories = []
    references = []
    standard_name = None
    long_name = None
    units = None
    attributes = {}
    cell_methods = []
    dim_coords_and_dims = []
    aux_coords_and_dims = []

    exclude = exclude or []

    if 'scalar_time_coords' not in exclude:
        aux_coords_and_dims.extend(scalar_time_coords(f.lbproc, f.lbtim,
                          f.lbcode, f.time_unit('hours'), f.t1, f.t2, f.lbft))

#    if 'scalar_vertical_coords' not in exclude:
#        aux_coords_and_dims.extend(scalar_time_coords(f.lbproc, f.lbtim,
#                          f.lbcode, f.time_unit('hours'), f.t1, f.t2, f.lbft))

    if 'scalar_realization_coords' not in exclude:
        aux_coords_and_dims.extend(scalar_realization_coords(f.lbrsvd[3]))

    if 'scalar_pseudo_level_coords' not in exclude:
        aux_coords_and_dims.extend(scalar_pseudo_level_coords(f.lbuser[4]))


    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2, 4]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 12 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 3 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('djf', long_name='season', units='no_unit'), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2, 4]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 3 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 6 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('mam', long_name='season', units='no_unit'), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2, 4]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 6 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 9 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('jja', long_name='season', units='no_unit'), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2, 4]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 9 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 12 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('son', long_name='season', units='no_unit'), None))

    if \
            (f.bdx != 0.0) and \
            (f.bdx != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 1):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, standard_name=f._x_coord_name(), units='degrees', circular=(f.lbhem in [0, 4]), coord_system=f.coord_system()), 1))

    if \
            (f.bdx != 0.0) and \
            (f.bdx != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 2):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, standard_name=f._x_coord_name(), units='degrees', circular=(f.lbhem in [0, 4]), coord_system=f.coord_system(), with_bounds=True), 1))

    if \
            (f.bdy != 0.0) and \
            (f.bdy != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 1):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzy, f.bdy, f.lbrow, standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system()), 0))

    if \
            (f.bdy != 0.0) and \
            (f.bdy != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 2):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzy, f.bdy, f.lbrow, standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system(), with_bounds=True), 0))

    if \
            (f.bdy == 0.0 or f.bdy == f.bmdi) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.iy == 10)):
        dim_coords_and_dims.append((DimCoord(f.y, standard_name=f._y_coord_name(), units='degrees', bounds=f.y_bounds, coord_system=f.coord_system()), 0))

    if \
            (f.bdx == 0.0 or f.bdx == f.bmdi) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix == 11)):
        dim_coords_and_dims.append((DimCoord(f.x, standard_name=f._x_coord_name(),  units='degrees', bounds=f.x_bounds, circular=(f.lbhem in [0, 4]), coord_system=f.coord_system()), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.iy == 4):
        dim_coords_and_dims.append((DimCoord(f.y, standard_name='depth', units='m', bounds=f.y_bounds, attributes={'positive': 'down'}), 0))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.ix == 10) and \
            (f.bdx != 0) and \
            (f.bdx != f.bmdi):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system()), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.iy == 1) and \
            (f.bdy == 0 or f.bdy == f.bmdi):
        dim_coords_and_dims.append((DimCoord(f.y, long_name='pressure', units='hPa', bounds=f.y_bounds), 0))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.ix == 1) and \
            (f.bdx == 0 or f.bdx == f.bmdi):
        dim_coords_and_dims.append((DimCoord(f.x, long_name='pressure', units='hPa', bounds=f.x_bounds), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.iy == 23):
        dim_coords_and_dims.append((DimCoord(f.y, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.y_bounds), 0))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.ix == 23):
        dim_coords_and_dims.append((DimCoord(f.x, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.x_bounds), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.ix == 13) and \
            (f.bdx != 0):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, long_name='site_number', units='1'), 1))

    if \
            (len(f.lbcode) == 5) and \
            (13 in [f.lbcode.ix, f.lbcode.iy]) and \
            (11 not in [f.lbcode.ix, f.lbcode.iy]) and \
            (hasattr(f, 'lower_x_domain')) and \
            (hasattr(f, 'upper_x_domain')) and \
            (all(f.lower_x_domain != -1.e+30)) and \
            (all(f.upper_x_domain != -1.e+30)):
        aux_coords_and_dims.append((AuxCoord((f.lower_x_domain + f.upper_x_domain) / 2.0, standard_name=f._x_coord_name(), units='degrees', bounds=np.array([f.lower_x_domain, f.upper_x_domain]).T, coord_system=f.coord_system()), 1 if f.lbcode.ix == 13 else 0))

    if \
            (len(f.lbcode) == 5) and \
            (13 in [f.lbcode.ix, f.lbcode.iy]) and \
            (10 not in [f.lbcode.ix, f.lbcode.iy]) and \
            (hasattr(f, 'lower_y_domain')) and \
            (hasattr(f, 'upper_y_domain')) and \
            (all(f.lower_y_domain != -1.e+30)) and \
            (all(f.upper_y_domain != -1.e+30)):
        aux_coords_and_dims.append((AuxCoord((f.lower_y_domain + f.upper_y_domain) / 2.0, standard_name=f._y_coord_name(), units='degrees', bounds=np.array([f.lower_y_domain, f.upper_y_domain]).T, coord_system=f.coord_system()), 1 if f.lbcode.ix == 13 else 0))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cell_methods.append(CellMethod("mean", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 3):
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib not in [2, 3]):
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cell_methods.append(CellMethod("minimum", coords="time"))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cell_methods.append(CellMethod("minimum", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib != 2):
        cell_methods.append(CellMethod("minimum", coords="time"))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cell_methods.append(CellMethod("maximum", coords="time"))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cell_methods.append(CellMethod("maximum", coords="time", intervals="%d hour" % f.lbtim.ia))



    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib != 2):
        cell_methods.append(CellMethod("maximum", coords="time"))


    if f.lbuser[6] == 1 and f.lbuser[3] == 5226:
        standard_name = "precipitation_amount"
        units = "kg m-2"

    if \
            (f.lbuser[6] == 2) and \
            (f.lbuser[3] == 101):
        standard_name = "sea_water_potential_temperature"
        units = "Celsius"

    if \
            ((f.lbsrce % 10000) == 1111) and \
            ((f.lbsrce / 10000) / 100.0 > 0):
        attributes['source'] = 'Data from Met Office Unified Model %4.2f' % ((f.lbsrce / 10000) / 100.0)

    if \
            ((f.lbsrce % 10000) == 1111) and \
            ((f.lbsrce / 10000) / 100.0 == 0):
        attributes['source'] = 'Data from Met Office Unified Model'

    if f.lbuser[6] != 0 or (f.lbuser[3] / 1000) != 0 or (f.lbuser[3] % 1000) != 0:
        attributes['STASH'] = f.stash

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 4205):
        standard_name = "mass_fraction_of_cloud_ice_in_air"
        units = "1"

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 4206):
        standard_name = "mass_fraction_of_cloud_liquid_water_in_air"
        units = "1"

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 30204):
        standard_name = "air_temperature"
        units = "K"

    if \
            (f.lbuser[6] == 4) and \
            (f.lbuser[3] == 6001):
        standard_name = "sea_surface_wave_significant_height"
        units = "m"

    if str(f.stash) in STASH_TO_CF:
        standard_name = STASH_TO_CF[str(f.stash)].standard_name
        units = STASH_TO_CF[str(f.stash)].units
        long_name = STASH_TO_CF[str(f.stash)].long_name

    if \
            (not f.stash.is_valid) and \
            (f.lbfc in LBFC_TO_CF):
        standard_name = LBFC_TO_CF[f.lbfc].standard_name
        units = LBFC_TO_CF[f.lbfc].units
        long_name = LBFC_TO_CF[f.lbfc].long_name

    if f.lbuser[3] == 33:
        references.append(ReferenceTarget('orography', None))

    if f.lbuser[3] == 409 or f.lbuser[3] == 1:
        references.append(ReferenceTarget('surface_air_pressure', None))

    return (factories, references, standard_name, long_name, units, attributes,
            cell_methods, dim_coords_and_dims, aux_coords_and_dims)
