import mock

import file_resuscitator as fr 


def test_file_handle():
    
    with open('__init__.py', 'r') as fh:
        print dir(fh)
        print `fh`
        rescue = fr.FileObjectResuscitator.from_handle(fh)

    print rescue
    with rescue.resucitated() as fh:
        print fh.readlines()


if __name__ == '__main__':
    test_file_handle()
        
