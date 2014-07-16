from iris.fileformats.pp import PPField2
from iris.fileformats.ff import FF2PP
import itertools
import numpy as np


import iris.fileformats.rules
from iris.fileformats.pp_rules import vector_realization_coords, vector_time_coords

import biggus
import copy

import netcdftime

from collections import namedtuple


from _structured_array_identification import GroupStructure


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
        self._structure = None

    @property
    def structure(self):
        if self._structure is None:
            component_arrays = self.element_vectors
            components = {}
            for name, array in component_arrays.items():
                if array.ndim == 1:
                    array = array
                else:
                    # Produce a linear combination of a date-time since numpy doesn't
                    # support unique over a specific axis. We don't care about the actual
                    # date-times at this point, so we just need to ensure that they are
                    # combined uniquely without collisions (so just using total seconds will work).
                    date_weights = np.cumprod([1, 60, 60, 24, 31, 366])[::-1]
    
                    # A length 5 datetime was not including seconds (PPField2).
                    if array.shape[-1] == 5:
                        date_weights = date_weights[:-1]
                    array = np.sum(array * date_weights, axis=-1)
                components[name] = array
            struct = GroupStructure.from_component_arrays(components)
            self._structure = struct
        return self._structure

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

    def optimal_structure(self, permitted_structures):
        """
        Return the optimal structure given all of the permitted structures.

        The optimal, in this case, is defined as that which produces the most
        number of non-trivial dimensions.

        May return an empty list if no structure can be identified.

        """
        if not permitted_structures:
            return []
        return max(permitted_structures,
                   key=lambda potential: (np.prod(struct.size
                                                  for (name, struct) in potential),
                                          len(potential)))

    def element_array_and_dims(self, shape):
        """
        Given a target structure, return a dictionary entry for each element
        mapping element name to a numpy array and associated dimensions for
        the array, given the target structure.

        """
        elements_and_dimensions = self.structure.build_arrays(shape,
                                                              self.element_vectors)

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

    def permitted_structures(self):
        return self.structure.possible_structures()

    def construct_cube(self, target_structure=None):
        permitted_structures = self.permitted_structures()
        if target_structure is None:
            target_structure = self.optimal_structure(permitted_structures)

        if not target_structure:
            new_dims = (len(self), )
        else:
            new_dims = tuple(struct.size for (name, struct) in target_structure)

        # Don't load any of the data, just get the shape from the biggus array.
        shape = new_dims + self.fields[0]._data.shape

        structures_with_dimensions = self.element_array_and_dims(new_dims)
        array_elements = set(structures_with_dimensions.keys())

        def gen_fields(fname):
            assert fname is None
            
            field = copy.deepcopy(self.fields[0])
#            field.data = np.empty(shape)
            
            
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


def diff_field(f1, f2, exclude=('data', 'lbegin', 't1', 't2')):
    """
    A useful function to print the differences between two fields.
    """
    if f1 != f2:
        print '\nField diff:'
        for name in dir(f1):
            if not name.startswith('_') and name not in exclude:
                v1 = getattr(f1, name, None)
                v2 = getattr(f2, name, None)
                if not callable(v1) and v1 != v2:
                    print '   {}: {}, {}'.format(name, v1, v2)


if __name__ == '__main__':
    if False:
        fname = '/data/local/dataZoo/FF/alyku.pp0'
        fname = '/data/local/itpe/delme/xjanpa.ph19930101'
        ff = FF2PP(fname)
        fields = list(ff)
    else:
        fname_template = '/data/local/dataZoo/PP/GloSea4/prodf_op_sfc_cam_11_201107*_01*.pp'
        from iris.fileformats.pp import load as load_pp
        from glob import glob
        fields = list(itertools.chain.from_iterable(load_pp(fname) for fname in sorted(glob(fname_template))))

    for i, field_group in enumerate(collate(fields)):
#        print field_group.faster_potential_structures
#        print field_group.permitted_structures()
#        print field_group.element_array_and_dims(field_group.optimal_structure(field_group.permitted_structures()))


#        structure = field_group.optimal_structure(field_group.permitted_structures())
#        shape = tuple(struct.array_structure.size for struct in structure)


        cube = field_group.construct_cube()
        print cube.summary(shorten=True)

        continue
    
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
