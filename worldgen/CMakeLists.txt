cmake_minimum_required(VERSION 3.2)
project(worldgen)

add_custom_target(worldgen_python_ext
    COMMAND ${Python_EXECUTABLE} setup_linux.py build_ext --inplace
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets/creatures/geometry/cpp_utils
    COMMENT "Building Python extension NURBSE"
)