cmake_minimum_required(VERSION 3.10)
project(AirDrums)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(AirDrums main.cpp)

# Link OpenCV libraries
target_link_libraries(AirDrums ${OpenCV_LIBS})
