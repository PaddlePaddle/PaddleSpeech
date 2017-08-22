from setuptools import setup, Extension
import glob
import platform
import os


def compile_test(header, library):
    dummy_path = os.path.join(os.path.dirname(__file__), "dummy")
    command = "bash -c \"g++ -include " + header + " -l" + library + " -x c++ - <<<'int main() {}' -o " + dummy_path + " >/dev/null 2>/dev/null && rm " + dummy_path + " 2>/dev/null\""
    return os.system(command) == 0


FILES = glob.glob('kenlm/util/*.cc') + glob.glob('kenlm/lm/*.cc') + glob.glob(
    'kenlm/util/double-conversion/*.cc')
FILES = [
    fn for fn in FILES if not (fn.endswith('main.cc') or fn.endswith('test.cc'))
]

LIBS = ['stdc++']
if platform.system() != 'Darwin':
    LIBS.append('rt')

ARGS = ['-O3', '-DNDEBUG', '-DKENLM_MAX_ORDER=6', '-std=c++11']

if compile_test('zlib.h', 'z'):
    ARGS.append('-DHAVE_ZLIB')
    LIBS.append('z')

if compile_test('bzlib.h', 'bz2'):
    ARGS.append('-DHAVE_BZLIB')
    LIBS.append('bz2')

if compile_test('lzma.h', 'lzma'):
    ARGS.append('-DHAVE_XZLIB')
    LIBS.append('lzma')

os.system('swig -python -c++ ./decoders.i')

ctc_beam_search_decoder_module = [
    Extension(
        name='_swig_decoders',
        sources=FILES + glob.glob('*.cxx') + glob.glob('*.cpp'),
        language='C++',
        include_dirs=['.', './kenlm', './openfst-1.6.3/src/include'],
        libraries=LIBS,
        extra_compile_args=ARGS)
]

setup(
    name='swig_decoders',
    version='0.1',
    description="""CTC decoders""",
    ext_modules=ctc_beam_search_decoder_module,
    py_modules=['swig_decoders'], )
