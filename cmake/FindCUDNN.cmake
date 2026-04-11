# FindCUDNN.cmake
# Locates the cuDNN library and sets:
#   CUDNN_FOUND
#   CUDNN_INCLUDE_DIR
#   CUDNN_LIBRARIES
#   CUDNN_VERSION

cmake_minimum_required(VERSION 3.22)

set(_cudnn_search_paths
    ${CUDNN_ROOT}
    $ENV{CUDNN_ROOT}
    $ENV{CUDNN_PATH}
    /usr/local/cuda
    /usr/local
    /usr
)

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    PATHS ${_cudnn_search_paths}
    PATH_SUFFIXES include
    DOC "cuDNN include directory"
)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    PATHS ${_cudnn_search_paths}
    PATH_SUFFIXES lib64 lib lib/x86_64-linux-gnu
    DOC "cuDNN library"
)

# Parse version from cudnn_version.h (cuDNN 8+) or cudnn.h
if(CUDNN_INCLUDE_DIR)
    if(EXISTS "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
        file(READ "${CUDNN_INCLUDE_DIR}/cudnn_version.h" _cudnn_version_h)
    else()
        file(READ "${CUDNN_INCLUDE_DIR}/cudnn.h" _cudnn_version_h)
    endif()
    string(REGEX MATCH "CUDNN_MAJOR ([0-9]+)" _ "${_cudnn_version_h}")
    set(_major ${CMAKE_MATCH_1})
    string(REGEX MATCH "CUDNN_MINOR ([0-9]+)" _ "${_cudnn_version_h}")
    set(_minor ${CMAKE_MATCH_1})
    string(REGEX MATCH "CUDNN_PATCHLEVEL ([0-9]+)" _ "${_cudnn_version_h}")
    set(_patch ${CMAKE_MATCH_1})
    set(CUDNN_VERSION "${_major}.${_minor}.${_patch}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
    VERSION_VAR   CUDNN_VERSION
)

if(CUDNN_FOUND)
    set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
    if(NOT TARGET CUDNN::cudnn)
        add_library(CUDNN::cudnn UNKNOWN IMPORTED)
        set_target_properties(CUDNN::cudnn PROPERTIES
            IMPORTED_LOCATION             "${CUDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
