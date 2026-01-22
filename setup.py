from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'Provides various CUDA-accelerated functionality for the foveated HTGS method.'

ENABLE_NVCC_LINEINFO = True  # set to True for profiling kernels with Nsight Compute (overhead is minimal)


module_root = Path(__file__).parent.relative_to(Path.cwd())
extension_name = module_root.absolute().name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings']]

all_sources = []
for module in cuda_modules:
    all_sources += glob(str(extension_root / module / 'src' / '**'/ '*.cpp'), recursive=True)
    all_sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)

base_sources = [str(extension_root / 'torch_bindings' / 'bindings.cpp')]
fast_inference_sources = [
    str(extension_root / 'torch_bindings' / 'bindings_benchmarking.cpp'),
    str(extension_root / 'rasterization' / 'src' / 'inference.cu'),
]
stereo_sources = [str(extension_root / 'torch_bindings' / 'bindings_stereo.cpp')]
for src in all_sources:
    if 'stereo' in src:
        stereo_sources.append(src)
    elif 'monocular' in src:
        fast_inference_sources.append(src)
        base_sources.append(src)
    elif 'fast_inference' in src:
        fast_inference_sources.append(src)
    elif 'inference' in src:
        base_sources.append(src)
    else:
        base_sources.append(src)
        fast_inference_sources.append(src)
        stereo_sources.append(src)

include_dirs = [str(extension_root.absolute() / 'utils')]
for module in cuda_modules:
    include_dirs.append(str(extension_root.absolute() / module / 'include'))

cxx_flags, nvcc_flags = ['--std=c++20'], ['-std=c++20']
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

benchmark_cxx_flags = ['-O3']
benchmark_nvcc_flags = ['-O3', '-use_fast_math']

base_extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=base_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }
)

fast_inference_extension = CUDAExtension(
    name=f'{extension_name}._C_benchmarking',
    sources=fast_inference_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags + benchmark_cxx_flags,
        'nvcc': nvcc_flags + benchmark_nvcc_flags
    }
)

stereo_extension = CUDAExtension(
    name=f'{extension_name}._C_stereo',
    sources=stereo_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags + benchmark_cxx_flags,
        'nvcc': nvcc_flags + benchmark_nvcc_flags
    }
)

setup(
    name=extension_name,
    author=__author__,
    packages=[extension_name],
    ext_modules=[base_extension, fast_inference_extension, stereo_extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
