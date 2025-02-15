import os
import subprocess

from config import kernels, sources, target
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

target = target.lower()

# Set environment variables
tk_root = os.getenv('THUNDERKITTENS_ROOT',
                    os.path.abspath(os.path.join(os.getcwd(), 'tk/')))
python_include = subprocess.check_output([
    'python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"
]).decode().strip()
torch_include = subprocess.check_output([
    'python', '-c',
    "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"
]).decode().strip()
print('st_attn root:', tk_root)
print('Python include:', python_include)
print('Torch include directories:', torch_include)

# CUDA flags
cuda_flags = [
    '-DNDEBUG', '-Xcompiler=-Wno-psabi', '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda', '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler', '--use_fast_math', '-std=c++20',
    '-O3', '-Xnvlink=--verbose', '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills', f'-I{tk_root}/include',
    f'-I{tk_root}/prototype', f'-I{python_include}', '-DTORCH_COMPILE'
] + torch_include.split()
cpp_flags = ['-std=c++20', '-O3']

if target == 'h100':
    cuda_flags.append('-DKITTENS_HOPPER')
    cuda_flags.append('-arch=sm_90a')
else:
    raise ValueError(f'Target {target} not supported')

source_files = ['st_attn.cpp']
for k in kernels:
    if target not in sources[k]['source_files']:
        raise KeyError(
            f'Target {target} not found in source files for kernel {k}')
    if isinstance(sources[k]['source_files'][target], list):
        source_files.extend(sources[k]['source_files'][target])
    else:
        source_files.append(sources[k]['source_files'][target])
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')

setup(name='st_attn',
      version="0.0.0",
      packages=find_packages(),
      ext_modules=[
          CUDAExtension('st_attn_cuda',
                        sources=source_files,
                        extra_compile_args={
                            'cxx': cpp_flags,
                            'nvcc': cuda_flags
                        },
                        libraries=['cuda'])
      ],
      cmdclass={'build_ext': BuildExtension})
