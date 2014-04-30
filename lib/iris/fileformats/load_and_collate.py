from iris.fileformats.pp import PPField2
from iris.fileformats.ff import FF2PP
import itertools

from collections import namedtuple
Structure = namedtuple('Structure', ['elements', 'sequence', 'stride'])


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
        fields - a list of PPField instances. As linstances must be of the
                 same type (i.e. must all be PPField3/PPField2)
        """
        self.stash = stash
        self.fields = fields
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

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def potential_structures(self):
        # Will return all varying dimensions, even non-viable ones.
        potential_structures = {}
        for i, (field, next_field) in enumerate(pairwise(self.fields)):
            for elements, (v1, v2) in self.identify_changes(field, next_field).items():
#                print i, 'v1:', elements, v1, v2
                # Store a list of [values_list, indices_list, complete, viable] pairs, which we update as we go along.
                # n.b. the first value from this set is *always* found at index 0.
                potential = potential_structures.setdefault(elements, [[v1], [0], False, True, None])
                values, indices, complete, viable, value_gen = potential
                if not viable:
                    continue
                if v2 not in values:
                    if complete:
                        potential[3] = viable = False
                        continue

                    values.append(v2)
                    indices.append(i + 1)
                else:
                    if not complete:
                        # Then complete it - no more values can be added after this point.
                        # Complete this potential - we've repeated some values.
                        potential[2] = complete = True
                        potential[4] = value_gen = itertools.cycle(values)
                    
                    # XXX Do some assertions that this potentially works. Only mark the ones
                    # that don't as "not viable", allowing exceptions to be raised later on.
                    expected = next(value_gen)
#                    print 'Checking viability of:', elements, v2, expected
                    if v2 != expected:
                        potential[3] = viable = False
                        # XXX Perhaps log the reason.
        
        # Remove the last element.
        potential_structures = {k: v[:-1] for k, v in potential_structures.items()}

        return potential_structures

    def structure(self):
        potential_structures = self.potential_structures()
        
        # Sort by the last index found - this will put the quickest changing dimensions last (right most).
        sorted_structures = sorted(potential_structures.items(), key=lambda pair: (pair[1][-1], pair[0]))
        
        
        import numpy as np
        
        result = []
        for elements, (values, indices, complete, viable) in sorted_structures:
            if viable or True:
                # XXX Who checks the stride is consistent?
                # XXX Cast to array? What about objects?
                stride = next(iter(np.diff(indices)), 0)
                result.append(Structure(elements, np.array(values), stride))
                
        return result

    def permitted_structures(self):
        total_len = len(self)
        
        possible = []
        
        structures = self.structure()
        filter_stride_len = lambda length: filter(lambda structure: structure.stride == length, structures)

        # XXX Stride 0 values?
        for structure in filter_stride_len(1):
            possible.append([structure])
        
        allowed_structures = []
        while possible:
            for potential in possible[:]:
                possible.remove(potential)
                
                import numpy as np
#                print 'SUM: ', sum([len(pot.sequence) for pot in potential]), np.product([len(pot.sequence) for pot in potential])
                next_stride = np.product([len(pot.sequence) for pot in potential])
#                print 'Next:', next_stride, 'Required length:', len(self)
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

    def identify_changes(self, field1, field2):
        changes = {}
        if field1.blev != field2.blev:
            changes['blev'] = [field1.blev, field2.blev]
        # XXX Do time properly
        if self._field_t1(field1) != self._field_t1(field2):
            changes['time'] = [self._field_t1(field1), self._field_t1(field2)]
        
        if self._field_t2(field1) != self._field_t2(field2):
            changes['time2'] = [self._field_t2(field1), self._field_t2(field2)]
        
        if field1.lbft != field2.lbft:
            changes['lbft'] = [field1.lbft, field2.lbft]
        
        if field1.lbuser[4] != field2.lbuser[4]:
            changes['lbplev'] = [field1.lbuser[4], field2.lbuser[4]]
        
        # XXX Remove?
        if field1.lbexp != field2.lbexp:
            changes['lbexp'] = [field1.lbexp, field2.lbexp]
        
        # Ensemble member
        if field1.lbrsvd[3] != field2.lbrsvd[3]:
            changes['lbrsvd[3]'] = [field1.lbrsvd[3], field2.lbrsvd[3]]
        
        return changes


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
        yield FieldGroup(stash, list(fields))


if __name__ == '__main__':
#    import subprocess
#    print subprocess.check_call(['/usr/local/sci/bin/nosetests',
#                                   'iris.tests.unit.fileformats.ff.test_load_and_collate:Test_FieldGroup.test_2d_structure_with_non_viable',
#                                   '-sv'])
#    exit()
    
    if True:
        fname = '/data/local/dataZoo/FF/alyku.pp0'
        fname = '/net/project/badc/hiresgw/xjanpa.ph19930101'
        fname = '/data/local/itpe/delme/xjanpa.ph19930101'
        ff = FF2PP(fname)
        fields = list(ff)
    else:
        fname_template = '/data/local/dataZoo/PP/GloSea4/prodf_op_sfc_cam_11_201107*_01*.pp'
        from iris.fileformats.pp import load as load_pp
        from glob import glob
        fields = list(itertools.chain.from_iterable(load_pp(fname) for fname in glob(fname_template)))

    def diff_field(f1, f2, exclude=('data', 'lbegin', 't1', 't2')):
        if f1 != f2:
            print '\nField diff:'
            for name in dir(f1):
                if not name.startswith('_') and name not in exclude:
                    v1 = getattr(f1, name, None)
                    v2 = getattr(f2, name, None)
                    if not callable(v1) and v1 != v2:
                        print '   {}: {}, {}'.format(name, v1, v2)
    exit()
    for field_group in collate(fields):
        print 'Possibles:'
        for structures in field_group.permitted_structures():
            print ' ' + '; '.join('{}: {}'.format(structure.elements, len(structure.sequence))
                                  for structure in structures)
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

