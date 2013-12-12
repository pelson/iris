from contextlib import contextmanager
import os


class FilelikeResuscitator(object):
    def recreated_filelike(self, *args, **kwargs):
        """Return the file-like - use resucitated for context manager - much better."""
        return self.resucitated(*args, **kwargs).__enter__()

    @contextmanager
    def resucitated(self):
        """A context manager for accessing the resuscitated file-like."""
        yield

    @classmethod
    def from_handle(cls, fh):
        """
        Given a file handle which this class can resuscitate, return the Resuscitator instance
        which can recreate a comparable file-like object.
        """
        return cls()
    
    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.__dict__ == other.__dict__
        else: 
            return NotImplemented
    
    def __ne__(self, other):
        return not self == other
    
    def __repr__(self):
        kwargs = ', '.join('{}={!r}'.format(key, value) for key, value in self.__dict__.items())
        return '{klass.__name__}({kwargs})'.format(klass=self.__class__, kwargs=kwargs)


class FileObjectResuscitator(FilelikeResuscitator):
    def __init__(self, name, mode, position):
        self.name, self.mode, self.position = name, mode, position
    
    @contextmanager
    def resucitated(self):
        # XXX: Perhaps use "io.open"?
        with open(self.name, self.mode) as fh:
            fh.seek(self.position, os.SEEK_SET)
            yield fh

    @classmethod
    def from_handle(cls, fh):
        return cls(fh.name, fh.mode, fh.tell())

# XXX Implement a SeekableFileHandleAdapter for non-seekable file handles. 


if __name__ == '__main__':
    import iris
    
    with open('/data/local/dataZoo/PP/aPPglob1/global.pp', 'rb') as fh:
        cube, = iris.load(fh)
    print cube._data_manager
    print cube._data
    print cube.data
#    
#    with open('__init__.py', 'r') as fh:
#        print dir(fh)
#        print `fh`
#        rescue = FileObjectResuscitator.from_handle(fh)
#
#    print rescue
#    with rescue.resucitated() as fh:
#        print fh.readlines()
#        
