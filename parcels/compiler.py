import os
from codepy.toolchain import guess_toolchain

def get_package_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))

def get_compiler():
    compiler = guess_toolchain()
    compiler.include_dirs += ['%s/include' % get_package_dir()]
    return compiler
