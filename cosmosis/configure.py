import os
import sys
from . import compilers

if '--debug' in sys.argv:
    common_flags = "-O3 -g -fPIC"


dirname = os.path.split(__file__)[0]
cosmosis_src_dir =  os.path.abspath(os.path.join(dirname, os.path.pardir))



commands = """

export COSMOSIS_SRC_DIR={cosmosis_src_dir}
export C_INCLUDE_PATH=$C_INCLUDE_PATH:{cosmosis_src_dir}/cosmosis/datablock
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH:{cosmosis_src_dir}/cosmosis/datablock
export LIBRARY_PATH=$LIBRARY_PATH:{cosmosis_src_dir}/cosmosis/datablock
export COSMOSIS_ALT_COMPILERS=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{cosmosis_src_dir}/cosmosis/datablock
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:{cosmosis_src_dir}/cosmosis/datablock
""".format(**locals())

commands += compilers.compilers

if __name__ == '__main__':
    print(commands)
