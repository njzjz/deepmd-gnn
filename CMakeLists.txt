# FindCUDAToolkit and execute_process(COMMAND_ERROR_IS_FATAL), used by the
# CUDA/dynamic-cudart build, require CMake 3.19 or newer.
cmake_minimum_required(VERSION 3.19)
project(deepmd-gnn CXX)

set(CMAKE_CXX_STANDARD 14)
macro(set_if_higher VARIABLE VALUE)
  # ${VARIABLE} is a variable name, not a string
  if(${VARIABLE} LESS "${VALUE}")
    set(${VARIABLE} ${VALUE})
  endif()
endmacro()

# build cpp or python interfaces
option(BUILD_CPP_IF "Build C++ interfaces" ON)
option(BUILD_PY_IF "Build Python interfaces" OFF)
option(USE_PT_PYTHON_LIBS "Use PyTorch Python libraries" OFF)
set(DEEPMD_GNN_WITH_CUDA
    "AUTO"
    CACHE STRING
          "Build CUDA implementation of deepmd-gnn OPs: AUTO, ON, or OFF")
set_property(CACHE DEEPMD_GNN_WITH_CUDA PROPERTY STRINGS AUTO ON OFF)
option(DEEPMD_GNN_BYPASS_TORCH_CUDA_CHECK
       "Bypass PyTorch CUDA toolkit discovery when nvcc is unavailable" ON)
option(DEEPMD_GNN_LINK_TORCH_CUDA "Link the OP against PyTorch CUDA libraries"
       OFF)
option(DEEPMD_GNN_DYNAMIC_CUDART
       "Dynamically load the CUDA runtime library for CUDA OPs" ON)

if((NOT BUILD_PY_IF) AND (NOT BUILD_CPP_IF))
  # nothing to do
  message(FATAL_ERROR "Nothing to build.")
endif()

string(TOUPPER "${DEEPMD_GNN_WITH_CUDA}" DEEPMD_GNN_WITH_CUDA)
if(NOT DEEPMD_GNN_WITH_CUDA MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "DEEPMD_GNN_WITH_CUDA must be AUTO, ON, or OFF")
endif()

if(DEEPMD_GNN_WITH_CUDA STREQUAL "ON")
  find_package(CUDAToolkit REQUIRED)
elseif(DEEPMD_GNN_WITH_CUDA STREQUAL "AUTO")
  find_package(CUDAToolkit QUIET)
endif()

if(CUDAToolkit_FOUND)
  set(DEEPMD_GNN_NVCC_EXECUTABLE ${CUDAToolkit_NVCC_EXECUTABLE})
else()
  set(DEEPMD_GNN_NVCC_EXECUTABLE "DEEPMD_GNN_NVCC_EXECUTABLE-NOTFOUND")
endif()

if(DEEPMD_GNN_WITH_CUDA STREQUAL "AUTO")
  if(CUDAToolkit_FOUND AND DEEPMD_GNN_NVCC_EXECUTABLE)
    set(DEEPMD_GNN_ENABLE_CUDA ON)
  else()
    set(DEEPMD_GNN_ENABLE_CUDA OFF)
  endif()
else()
  set(DEEPMD_GNN_ENABLE_CUDA ${DEEPMD_GNN_WITH_CUDA})
endif()

if(DEEPMD_GNN_ENABLE_CUDA)
  if(NOT CUDAToolkit_FOUND OR NOT DEEPMD_GNN_NVCC_EXECUTABLE)
    message(FATAL_ERROR "DEEPMD_GNN_WITH_CUDA=ON requires CUDAToolkit/nvcc")
  endif()
  message(
    STATUS
      "Found CUDA in ${CUDAToolkit_BIN_DIR}; enabling CUDA OP with ${DEEPMD_GNN_NVCC_EXECUTABLE}"
  )
  set(DEEPMD_GNN_BYPASS_TORCH_CUDA_CHECK
      OFF
      CACHE BOOL
            "Bypass PyTorch CUDA toolkit discovery when nvcc is unavailable"
            FORCE)
  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER ${DEEPMD_GNN_NVCC_EXECUTABLE})
  endif()
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
  enable_language(CUDA)
else()
  message(
    STATUS "CUDAToolkit/nvcc not found or CUDA disabled; building CPU-only OP")
endif()

if(BUILD_CPP_IF
   AND USE_PT_PYTHON_LIBS
   AND NOT CMAKE_CROSSCOMPILING
   AND NOT SKBUILD
   OR "$ENV{CIBUILDWHEEL}" STREQUAL "1")
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import torch;print(torch.utils.cmake_prefix_path)"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH
    RESULT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR
    ERROR_VARIABLE PYTORCH_CMAKE_PREFIX_PATH_ERROR_VAR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ${PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR} EQUAL 0)
    message(
      FATAL_ERROR
        "Cannot determine PyTorch CMake prefix path, error code: $PYTORCH_CMAKE_PREFIX_PATH_RESULT_VAR}, error message: ${PYTORCH_CMAKE_PREFIX_PATH_ERROR_VAR}"
    )
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})
endif()

if(DEEPMD_GNN_BYPASS_TORCH_CUDA_CHECK)
  if(NOT DEEPMD_GNN_NVCC_EXECUTABLE AND NOT TARGET torch::cudart)
    message(
      STATUS
        "nvcc not found; bypassing PyTorch CUDA toolkit discovery for this CPU-only OP"
    )
    add_library(torch::cudart INTERFACE IMPORTED)
  endif()
endif()
find_package(Torch REQUIRED)
if(Torch_VERSION VERSION_LESS "2.10.0")
  message(FATAL_ERROR "deepmd-gnn OP requires PyTorch >= 2.10.0 for the "
                      "LibTorch Stable ABI.")
endif()
set(DEEPMD_GNN_TORCH_LIBRARIES ${TORCH_LIBRARIES})
if((NOT DEEPMD_GNN_LINK_TORCH_CUDA) AND TARGET torch_cpu)
  set(DEEPMD_GNN_TORCH_LIBRARIES torch_cpu)
  message(STATUS "Linking deepmd-gnn OP against PyTorch CPU runtime")
endif()
if(NOT Torch_VERSION VERSION_LESS "2.1.0")
  set_if_higher(CMAKE_CXX_STANDARD 17)
elseif(NOT Torch_VERSION VERSION_LESS "1.5.0")
  set_if_higher(CMAKE_CXX_STANDARD 14)
endif()

# define build type
if((NOT DEFINED CMAKE_BUILD_TYPE) OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE release)
endif()

add_subdirectory(op)
