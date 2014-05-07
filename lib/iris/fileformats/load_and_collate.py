from iris.fileformats.pp import PPField2
from iris.fileformats.ff import FF2PP
import itertools
import numpy as np

import netcdftime

from collections import namedtuple

ElementStructure = namedtuple('ElementStructure', ['elements', 'array_structure'])


class ArrayStructure(namedtuple('ArrayStructure',
                                ['stride', 'unique_ordered_values', 'size'])):
    def __eq__(self, other):
        stride = getattr(other, 'stride', None)
        arr = getattr(other, 'unique_ordered_values', None)
        
        if stride is None or arr is None:
            return NotImplemented
        else:
            return (stride == self.stride and
                    np.all(self.unique_ordered_values == arr))

    def __ne__(self, other):
        return not (self == other)


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


class FieldGroup(object):
    """
    Represents fields grouped by STASH. This class can be used to identify
    structure of the group (e.g. height and time dimensionality).

    """
    def __init__(self, stash, fields):
        """
        Kwargs
        ------
        stash - a tuple representing the raw STASH components of a field
                (i.e. ``self.lbuser[6], self.lbuser[3]``)
        fields - a iterable of PPField instances. All instances must be of the
                 same type (i.e. must all be PPField3/PPField2).
                 Fields are immutable.
        """
        self.stash = stash
        self._fields = tuple(fields)
        assert len(self._fields) > 0

        if isinstance(fields[0], PPField2):
            self._field_t1 = lambda field: (field.lbyr, field.lbmon, field.lbdat,
                                            field.lbhr, field.lbmin)
            self._field_t2 = lambda field: (field.lbyrd, field.lbmond, field.lbdatd,
                                            field.lbhrd, field.lbmind)
        else:
            self._field_t1 = lambda field: (field.lbyr, field.lbmon, field.lbdat,
                                            field.lbhr, field.lbmin, field.lbsec)
            self._field_t2 = lambda field: (field.lbyrd, field.lbmond, field.lbdatd,
                                            field.lbhrd, field.lbmind, field.lbsecd)

        #: A dictionary mapping element name to a numpy array of
        #: element values (scalars for simple elements, multi-dimensional arrays
        #: for those such as datetimes etc.).
        self._element_vectors = None
        self._unstructured_elements = self._scalar_elements = self._potential_ndarray_elements = None

    @property
    def fields(self):
        return self._fields

    def __len__(self):
        return len(self.fields)

    @property
    def element_vectors(self):
        """
        TODO.
        """
        if self._element_vectors is None:    
            self._element_vectors = {}
            for name, getter in {'lbft': lambda f: f.lbft,
                                 'time': self._field_t1,
                                 'time2': self._field_t2,
                                 'lbrsvd[3]': lambda f: f.lbrsvd[3],
                                 'blev': lambda f: f.blev}.items():
                self._element_vectors[name] = np.array([getter(field) for field in self.fields])
        return self._element_vectors 

    @property
    def faster_potential_structures(self):
        """
        Compute ...
        """
        if self._unstructured_elements is None:
            self._unstructured_elements = []
            self._scalar_elements = []
            self._potential_ndarray_elements = []

            for element, values in self.element_vectors.items():
                if values.ndim == 1:
                    combined_values = values
                else:
                    # Produce a linear combination of a date-time since numpy doesn't
                    # support unique over a specific axis. We don't care about the actual
                    # date-times at this point, so we just need to ensure that they are
                    # combined uniquely without collisions (so just using total seconds will work).
                    date_weights = np.cumprod([1, 60, 60, 24, 31, 366])[::-1]
                    
                    # A length 5 datetime was not including seconds (PPField2).
                    if values.shape[-1] == 5:
                        date_weights = date_weights[:-1]
    
                    combined_values = np.sum(values * date_weights, axis=-1)
    
                structure = find_structure(combined_values)

                if structure is None:
                    self._unstructured_elements.append(element)
                elif structure.unique_ordered_values.size == 1:
                    self._scalar_elements.append(ElementStructure(element, structure))
                else:
                    self._potential_ndarray_elements.append(ElementStructure(element, structure))

        return self._unstructured_elements, self._scalar_elements, self._potential_ndarray_elements

    def permitted_structures(self):
        # Must return something currently...
        possible = []
        _, _, vector_structures = self.faster_potential_structures
        filter_stride_len = lambda length: [struct for struct in vector_structures
                                            if struct.array_structure.stride == length]

        # XXX Stride 0 values?
        for structure in filter_stride_len(1):
            possible.append([structure])
        
        allowed_structures = []
        while possible:
            for potential in possible[:]:
                possible.remove(potential)
                next_stride = np.product([len(pot.array_structure.unique_ordered_values) for pot in potential])
                # If we've found a structure whose product is the length of the fields
                # of this Group, we've got a valid potential.
                if next_stride == len(self):
                    allowed_structures.append(potential)

                next_structs = filter_stride_len(next_stride)
                
                # If this can't be done (i.e. there isn't a dimension which would match up
                # with the current potential) then it wont get added back to the possibles.
                for struct in next_structs:
                    new_potential = potential[:]
                    new_potential.append(struct)
                    possible.append(new_potential)

        return allowed_structures

    def optimal_structure(self, permitted_structures):
        """
        Return the optimal structure given all of the permitted structures.
        
        The optimal, in this case, is defined as that which produces the most
        number of non-trivial dimensions.
        
        May return an empty list if no structure can be identified.

        """
        if not permitted_structures:
            return []
        return max(permitted_structures, key=lambda structure: (np.prod(struc.array_structure.size for struc in structure), len(structure)))

    def element_array_and_dims(self, shape):
        """
        Given a target structure, return a dictionary entry for each element
        mapping element name to a numpy array and associated dimensions for
        the array, given the target structure.

        """
#        shape = tuple(struct.array_structure.size for struct in target_structure)
        elements_and_dimensions = {}

        unstructured_names, scalars, vector_potentials = self.faster_potential_structures

        for struct in vector_potentials:
            stride_product = 1
            for dim, length in enumerate(shape):
                if struct.array_structure.stride == stride_product and length == struct.array_structure.size:
                    vector = self.element_vectors[struct.elements].reshape(shape + (-1, ))
                    # XXX Use the array elements array?
                    # Reduce the dimensionality to a 1d array.
                    vector = vector[tuple(0 if dim != i else slice(None)
                                          for i in range(len(shape)))]
                    elements_and_dimensions[struct.elements] = [vector, [dim]]
#                    print 'Bar:', shape, self.element_vectors[struct.elements].shape, struct.elements, vector.shape
                    break
                stride_product *= length
            else:
#                print 'GOING FOR IT:', self.element_vectors[struct.elements].shape, len(self)
                unstructured_names.append(struct.elements)
#                pass
#                raise ValueError('Should becomes an unstructured array.')

        for name in unstructured_names:
            # XXX Check the order - it was necessary to fix up the multi-dimensional coordinate construction.
#            print shape, shape + (-1, )
#            print 'Inside:', target_structure, len(self), self.element_vectors[name].shape, self.element_vectors[name].reshape(shape + (-1, )).shape
#            print vector_potentials
            elements_and_dimensions[name] = [self.element_vectors[name].reshape(shape + (-1, ), order='F'),
                                             range(len(shape))]

        for struct in scalars:
            elements_and_dimensions[struct.elements] = [self.element_vectors[struct.elements][:1], ()]

        for pair in elements_and_dimensions.items():
            arr, dims = pair[1]
            if arr.ndim > (len(dims) or 1):
                # XXX Generalise the collapsing of the second dimension so that it is easily extensible.
                arr_shape = arr.shape[:-1]
                extra_length = arr.shape[-1]
                if extra_length in [5, 6]:
                    # Flatten out the array apart from the last dimension,
                    # convert to netcdftime objects, then reshape back.
                    arr = np.array([netcdftime.datetime(*args)
                                    for args in arr.reshape(-1, extra_length)]
                                   ).reshape(arr_shape)
                else:
                    assert arr.shape[-1] == 1, 'Unexpected array dimensionality {}'.format(arr.shape)
                    arr = arr[..., 0]
                pair[1][0] = arr

        return elements_and_dimensions

    def construct_cube(self, target_structure=None):
        permitted_structures = self.permitted_structures()
        if target_structure is None:
            target_structure = self.optimal_structure(permitted_structures)

        if not target_structure:
            new_dims = (len(self), )
        else:
            new_dims = tuple(structure.array_structure.size for structure in target_structure)

        # Don't load any of the data, just get the shape from the biggus array.
        shape = new_dims + self.fields[0]._data.shape
        
        structures_with_dimensions = self.element_array_and_dims(new_dims)
        array_elements = set(structures_with_dimensions.keys())

        def gen_fields(fname):
            assert fname is None
            import copy
            field = copy.deepcopy(self.fields[0])
#            field.data = np.empty(shape)
            import biggus
            
            def groups_of(length, total_length):
                indices = tuple(range(0, total_length, length)) + (None, )
                return pairwise(indices)
    
            # XXX Todo: Verify that the field order is being interpreted correctly. 
            next_dim_arrays = [f._data for f in self.fields]
            for length in new_dims[::-1]:
                this_dim_arrays = next_dim_arrays
                next_dim_arrays = []
                for start, stop in groups_of(length, len(this_dim_arrays)):
                    sub = biggus.ArrayStack(np.array(this_dim_arrays[start:stop], dtype=object))
                    next_dim_arrays.append(sub)
            else:
                field._data = next_dim_arrays[0]

            # Modify the field.
            yield field

        import iris.fileformats.rules
        from iris.fileformats.pp_rules import vector_realization_coords, vector_time_coords
        def construct_vector_coords(field):
            f = field
            coords_and_dims = []
            # XXX Use scalar information?

            # Time vector coords.
            if set(['lbft', 'time', 'time2']).intersection(array_elements):
                t1, t1_dims = structures_with_dimensions['time']
                t2, t2_dims = structures_with_dimensions['time2']
                lbft, lbft_dims = structures_with_dimensions['lbft']

#                if 'time' not in array_elements:
#                    t1, t1_dims = field.t1, ()
#
#                if 'time2' not in array_elements:
#                    t2, t2_dims = field.t2, ()
#                
#                if 'lbft' not in array_elements:
#                    lbft, lbft_dims = field.lbft, ()
#                print structures_with_dimensions
                coords_and_dims.extend(vector_time_coords(f.lbproc, f.lbtim, f.lbcode, f.time_unit('hours'),
                                                     [t1, t1_dims],
                                                     [t2, t2_dims],
                                                     [lbft, lbft_dims]))

            if set(['lbrsvd[3]']).intersection(array_elements):
                # XXX Allow this to become a DimCoord
                lbrsvd3, dims = structures_with_dimensions['lbrsvd[3]']
                coords_and_dims.extend(vector_realization_coords([lbrsvd3, dims]))

            return coords_and_dims

        elements_to_exclude_names = {'lbrsvd[3]': 'scalar_realization_coords',
                                     'lbft': 'scalar_time_coords',
                                     'time': 'scalar_time_coords',
                                     'time2': 'scalar_time_coords',
                                     'blev': 'scalar_vertical_coords'}
        
        exclude = []
        for element in array_elements:
            exclude.append(elements_to_exclude_names[element])
        
        def converter(field):
            result = list(iris.fileformats.pp_rules.convert(field, exclude=exclude))
            dim_coords_and_dims, aux_coords_and_dims = result[-2:]
            dim_coords_and_dims = map(list, dim_coords_and_dims)
            aux_coords_and_dims = map(list, aux_coords_and_dims)
            for coord_dim_pair in itertools.chain(dim_coords_and_dims, aux_coords_and_dims):
                coord, dims = coord_dim_pair
                if dims is not None:
                    coord_dim_pair[1] += len(new_dims)

            # Construct the new vector coordinates. 
            aux_coords_and_dims.extend(construct_vector_coords(field))

            result[-2:] = dim_coords_and_dims, aux_coords_and_dims 
            return result

        # XXX TODO - the loader needs to move down out of the FieldGroup to give context of all other
        # fields in the file (such as reference orography etc.).
        pp_loader = iris.fileformats.rules.Loader(gen_fields, {},
                                                  converter, None)
        cubes = iris.fileformats.rules.load_cubes(filenames=[None], user_callback=None,
                                                  loader=pp_loader)
        cubes = list(cubes)
        assert len(cubes) == 1
        return cubes[0]


def collate(fields):

    # XXX Should this include the grid location? (i.e. bdx, bdy, bzy, bzx, x, y,
    #                                             x_bounds, y_bounds)
    stash_and_friends_tuple = lambda field: ((field.lbuser[6], field.lbuser[3]),
                                             (field.lbtim, field.lbrow, field.lbnpt,
                                              field.lbproc))
    # TODO: Expose sort as a keyword.
    fields = sorted(fields, key=stash_and_friends_tuple)
    # Everything we don't group by (like the y and x coordinates) could be a headache...
    for (stash, _), fields in itertools.groupby(fields, stash_and_friends_tuple):
        yield FieldGroup(stash, tuple(fields))


def find_structure(combined_values):
    # Note: This algorithm will only find distinct value columns/rows/axes any dimension
    # with repeating values will not have its structure identified and will be considered
    # irregular.
    if combined_values.size == 0:
        return ArrayStructure(1, combined_values, 0)
    
    # unique is a *sorted* array of unique values.
    # unique_inds takes us from the sorted unique values back to inds in the input array
    # inds_back_to_orig gives us the indices of each value in the array vs the index in the
    # *sorted* unique array.
    _, unique_inds = np.unique(combined_values, return_index=True)
    
    # Return the unique values back into a sorted array.
    unique = combined_values[np.sort(unique_inds)]

    # what we actually want is inds_back_to_orig in the sort order of the original array.
    new_inds = np.empty(combined_values.shape, dtype=unique_inds.dtype)
    new_inds.fill(-1)
    
    for ind, unique_val in enumerate(unique):
        new_inds[combined_values == unique_val] = ind
    assert np.all(new_inds > -1)

    inds_back_to_orig = new_inds 
    
    u_len = len(unique)
    n_fields = combined_values.size
    
    structure = None
    
    # If the length of the unique values is not a divisor of the
    # length of the original array, it is going to be an irregular
    # array, so we can avoid some processing.
    if (n_fields // u_len) != (float(n_fields) // u_len):
        # No structure.
        pass
    # Shortcut the simple case of all values being distinct.
    elif u_len == 1:
        structure = ArrayStructure(1, unique, u_len)
    else:
        # Working in index space, compute indices where values change.
        ind_diffs = np.diff(inds_back_to_orig)

        # Find the indices where a change takes place.
        stride_inds = np.nonzero(ind_diffs)
        
        # The first change (in index space) must be the stride length.
        # If we have no changes throughout the array, we can make the
        # stride length 1. 
        try:
            stride = np.diff(stride_inds[0])[0]
        except IndexError:
            stride = 1

        # Assert that all values make an equidistant change (in index space),
        # else the array is irregular.
        ind_diffs_non_zero = ind_diffs[stride_inds]
        if not np.all(np.diff(stride_inds) == stride):
            return None

        # Assert that each time we iterate over the length of the unique
        # values, we jump back to the first unique value (remember we are in
        # index space).
        jump_spots = ind_diffs_non_zero[(u_len-1)::u_len]
        if not np.all(jump_spots == (-u_len + 1) * stride):
            return None

        structure = ArrayStructure(stride, unique, u_len)

    return structure


if __name__ == '__main__':
#    import subprocess
#    print subprocess.check_call(['/usr/local/sci/bin/nosetests',
#                                   'iris.tests.unit.fileformats.ff.test_load_and_collate:Test_FieldGroup.test_2d_structure_with_non_viable',
#                                   '-sv'])
#    exit()
    
    if True:
        fname = '/data/local/dataZoo/FF/alyku.pp0'
#        fname = '/net/project/badc/hiresgw/xjanpa.ph19930101'
        fname = '/data/local/itpe/delme/xjanpa.ph19930101'
        ff = FF2PP(fname)
        fields = list(ff)
    else:
        fname_template = '/data/local/dataZoo/PP/GloSea4/prodf_op_sfc_cam_11_201107*_01*.pp'
        from iris.fileformats.pp import load as load_pp
        from glob import glob
        fields = list(itertools.chain.from_iterable(load_pp(fname) for fname in sorted(glob(fname_template))))

    def diff_field(f1, f2, exclude=('data', 'lbegin', 't1', 't2')):
        if f1 != f2:
            print '\nField diff:'
            for name in dir(f1):
                if not name.startswith('_') and name not in exclude:
                    v1 = getattr(f1, name, None)
                    v2 = getattr(f2, name, None)
                    if not callable(v1) and v1 != v2:
                        print '   {}: {}, {}'.format(name, v1, v2)

    for i, field_group in enumerate(collate(fields)):
#        print field_group.faster_potential_structures
#        print field_group.permitted_structures()
#        print field_group.element_array_and_dims(field_group.optimal_structure(field_group.permitted_structures()))

        structure = field_group.optimal_structure(field_group.permitted_structures())
        shape = tuple(struct.array_structure.size for struct in structure)

#        if i != 101:
#            continue
        cube = field_group.construct_cube()
#        if len(field_group) == 0:
        print cube.summary(shorten=True)
        print cube
#        print c.coord('forecast_period')
#        break
        continue
    
#        if field_group.stash != (1, 3236):
#            continue
        
        struct = field_group.structure()
        if len(field_group) > 1:
            if not struct:
                print 'Not found.', len(field_group)
                diff_field(field_group.fields[0], field_group.fields[1])
            else:
                print field_group.stash, len(struct[0].sequence), len(field_group), struct
                if len(struct[0].sequence) != len(field_group):
                    last_index = len(struct[0].sequence) * struct[0].stride - 1
                    try:
                        diff_field(field_group.fields[last_index], field_group.fields[last_index + 1])
                    except IndexError:
                        pass
