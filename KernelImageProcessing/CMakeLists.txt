cmake_minimum_required(VERSION 3.18)
project(KernelImageProcessing LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set policy for CUDA
cmake_policy(SET CMP0146 OLD)

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

# CUDA toolkit
find_package(CUDA REQUIRED)

# Sequential library
add_library(SequentialConvolution STATIC
    src/CpuConvolution.cpp
    src/Kernels.cpp
)
target_include_directories(SequentialConvolution PUBLIC include)
target_include_directories(SequentialConvolution PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SequentialConvolution PUBLIC ${OpenCV_LIBS})

# GPU library
add_library(GpuConvolution STATIC
    src/GpuConvolution.cu
)
target_include_directories(GpuConvolution PUBLIC include)
target_include_directories(GpuConvolution PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(GpuConvolution PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(GpuConvolution PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES})

set_target_properties(GpuConvolution PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Executable
add_executable(KernelApp src/main.cpp)
target_include_directories(KernelApp PRIVATE include)
target_link_libraries(KernelApp PRIVATE SequentialConvolution GpuConvolution )

# Force link OpenEXR and Imath
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENEXR REQUIRED OpenEXR)
pkg_check_modules(IMATH REQUIRED Imath)

target_link_libraries(KernelApp PRIVATE ${OPENEXR_LIBRARIES} ${IMATH_LIBRARIES})
target_link_directories(KernelApp PRIVATE ${OPENEXR_LIBRARY_DIRS} ${IMATH_LIBRARY_DIRS})

# Add include directories for OpenEXR
target_include_directories(KernelApp PRIVATE ${OPENEXR_INCLUDE_DIRS} ${IMATH_INCLUDE_DIRS})
