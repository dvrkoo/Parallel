cmake_minimum_required(VERSION 3.10)
project(KMeans)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include the include directory
include_directories(include)

# Add the src directory
add_subdirectory(src)
