# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Information_Optimisation)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find the YAML-CPP library (for YAML support)
find_package(yaml-cpp REQUIRED)

# Add the executable
add_executable(main src/main.cpp)

# Link OpenCV and yaml-cpp libraries to your project
target_link_libraries(main ${OpenCV_LIBS} yaml-cpp)

# Include OpenCV header files
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
