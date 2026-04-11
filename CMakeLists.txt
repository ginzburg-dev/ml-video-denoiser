cmake_minimum_required(VERSION 3.22)

project(denoiser
    LANGUAGES CXX CUDA
    VERSION 0.1.0
    DESCRIPTION "ML video denoiser — C++ inference engine"
)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# NVCC does not fully support C++23 yet — use C++20 for device code
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# CUDA architectures — Turing fallback + Ampere A100 + Ampere RTX 30-series
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING
    "CUDA architectures to compile for (sm_75 Turing, sm_80 A100, sm_86 RTX 30xx)")

# ---------------------------------------------------------------------------
# Build type defaults
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDNN REQUIRED)

if(CUDNN_VERSION VERSION_LESS "8.5")
    message(WARNING
        "cuDNN ${CUDNN_VERSION} detected — cudnnResampleForward requires >= 8.5. "
        "Falling back to custom bilinear upsample kernel.")
    set(USE_CUDNN_RESAMPLE OFF)
else()
    set(USE_CUDNN_RESAMPLE ON)
endif()

# Optional: LibRaw for reading camera RAW files in the CLI
option(USE_LIBRAW "Enable LibRaw for camera RAW (NEF/CR2/…) reading" OFF)
if(USE_LIBRAW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBRAW REQUIRED libraw)
endif()

# ---------------------------------------------------------------------------
# Third-party header-only libraries (submodules)
# ---------------------------------------------------------------------------
add_subdirectory(third_party/nlohmann_json EXCLUDE_FROM_ALL)

# stb — header-only; expose as interface target
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE third_party/stb)

# tinyexr — header-only (single header + miniz amalgamation)
add_library(tinyexr INTERFACE)
target_include_directories(tinyexr INTERFACE third_party/tinyexr)

# ---------------------------------------------------------------------------
# denoiser_engine static library
# ---------------------------------------------------------------------------
add_library(denoiser_engine STATIC
    src/tensor.cu
    src/weight_loader.cc
    src/layers/conv2d.cu
    src/layers/batchnorm2d.cu
    src/layers/relu.cu
    src/layers/upsample.cu
    src/layers/maxpool2d.cu
    src/layers/deform_conv2d.cu
    src/kernels/model_kernels.cu
    src/models/nef_residual.cu
    src/models/nef_temporal.cu
    src/io/image_io.cc
    src/io/exr_io.cc
    src/io/video_io.cc
)

target_include_directories(denoiser_engine PUBLIC
    include
    third_party
    third_party/tinyexr
    ${CUDNN_INCLUDE_DIR}
)

target_link_libraries(denoiser_engine PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDNN::cudnn
    nlohmann_json::nlohmann_json
    stb
    tinyexr
)

if(USE_LIBRAW)
    target_link_libraries(denoiser_engine PUBLIC ${LIBRAW_LIBRARIES})
    target_include_directories(denoiser_engine PUBLIC ${LIBRAW_INCLUDE_DIRS})
    target_compile_definitions(denoiser_engine PUBLIC DENOISER_USE_LIBRAW)
endif()

target_compile_definitions(denoiser_engine PRIVATE
    $<$<BOOL:${USE_CUDNN_RESAMPLE}>:DENOISER_USE_CUDNN_RESAMPLE>
)

# Compiler flags
target_compile_options(denoiser_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --ptxas-options=-v
        -Xcompiler -Wall,-Wextra
    >
)

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(nef_denoise cli/main.cc)
target_link_libraries(nef_denoise PRIVATE denoiser_engine)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
option(DENOISER_BUILD_TESTS "Build C++ unit tests" ON)
if(DENOISER_BUILD_TESTS)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS nef_denoise RUNTIME DESTINATION bin)
install(DIRECTORY include/denoiser DESTINATION include)
