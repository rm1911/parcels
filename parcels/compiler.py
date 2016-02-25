from os import path
from codepy.toolchain import guess_toolchain


def get_parcels_dir():
    return path.abspath(path.join(path.dirname(__file__), path.pardir))


def get_codepy_dir():
    import codepy
    return path.abspath(path.join(path.dirname(codepy.__file__)))


def get_compiler():
    compiler = guess_toolchain()
    compiler.add_library('parcels', [path.join(get_parcels_dir(), 'include')], [], [])
    compiler.add_library('codepy', [path.join(get_codepy_dir(), 'include')], [], [])
    return compiler
