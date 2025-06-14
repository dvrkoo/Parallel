# Add the executable
add_executable(KMeans main.cpp SequentialKMeans.cpp ParallelKMeans.cpp)

# Find and link OpenMP if needed
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(KMeans PUBLIC OpenMP::OpenMP_CXX)
endif()
