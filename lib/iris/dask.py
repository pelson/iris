from __future__ import absolute_import

import os

import dask
from dask.delayed import tokenize
import iris


class dask_iris(object):
    merge = dask.delayed(iris.cube.CubeList.merge, pure=True)
    merge_cube = dask.delayed(iris.cube.CubeList.merge_cube, pure=True)
    concatenate = dask.delayed(iris.cube.CubeList.concatenate, pure=True)
    _load_raw = dask.delayed(iris.load_raw, pure=True)

    @classmethod
    def load(cls, fnames, *args, **kwargs):
        """
        Return a delayed CubeList from the given filenames.

        """
        return cls.merge(cls.load_raw(fnames, *args, **kwargs))

    @classmethod
    def load_cube(cls, fnames, *args, **kwargs):
        """
        Return a delayed CubeList from the given filenames.

        """
        return cls.merge_cube(cls.load_raw(fnames, *args, **kwargs))
    
    @classmethod
    def load_raw(cls, fnames, *args, **kwargs):
        """Return a delayed CubeList of raw cubes from the given files."""
        
        
        token_kwargs = kwargs.copy()
        token_kwargs.setdefault('pure', True)
        
        delayed_cube_lists = []
        for fname in fnames:
            # Try to put a pretty(ish) name to the load input.
            
            tokenized = tokenize(fname, *args, **token_kwargs)
            name = '{} ({})'.format(os.path.basename(fname), tokenized)
            kwargs['dask_key_name'] = name
            cubes = cls._load_raw(fname, *args, **kwargs)
            delayed_cube_lists.append(cubes)
        return cls.combine_results(delayed_cube_lists)
    
    @staticmethod
    @dask.delayed(pure=True)
    def combine_results(cube_lists):
        """Return a CubeList from a list of CubeLists."""
        cubes = iris.cube.CubeList()
        for cube_list in cube_lists:
            cubes.extend(cube_list)
        return cubes
