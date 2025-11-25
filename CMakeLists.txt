CMAKE_MINIMUM_REQUIRED(VERSION 3.27.2)

project(FoveatedHTGSRasterizer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_STANDARD 20)

file(GLOB SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/include/hybrid_blend/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/include/hybrid_blend/kernels/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/include/kernel_utils.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/include/rasterization_utils.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/include/shared_kernels.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/src/hybrid_blend/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/src/shared_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/utils/*.h"
)
add_library(FoveatedHTGSRasterizer STATIC ${SOURCES})

set(FoveatedHTGSRasterizer_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend
    PARENT_SCOPE
)
target_include_directories(FoveatedHTGSRasterizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/rasterization/include
    ${CMAKE_CURRENT_SOURCE_DIR}/FoveatedHTGSCudaBackend/utils
)
set_target_properties(FoveatedHTGSRasterizer PROPERTIES CUDA_ARCHITECTURES "86")
set_target_properties(FoveatedHTGSRasterizer PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(FoveatedHTGSRasterizer PROPERTIES FOLDER "extlibs")

target_compile_options(FoveatedHTGSRasterizer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --generate-line-info
  --use_fast_math
>)
