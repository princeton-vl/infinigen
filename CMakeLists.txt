cmake_minimum_required(VERSION 3.2)
project(infinigen)

set(CONDA_DIR $ENV{CONDA_PREFIX})
find_program(Python_EXECUTABLE python PATHS ${CONDA_DIR} ~/desktop/.conda NO_DEFAULT_PATH)
if(NOT Python_EXECUTABLE)
    message(FATAL_ERROR "Python executable not found in conda environment")
endif()
message(STATUS "Python executable found at ${Python_EXECUTABLE}")
get_filename_component(CONDA_ROOT ${Python_EXECUTABLE} DIRECTORY)
set(CONDA_INDICATOR_DIRECTORY ${CONDA_ROOT}/Library/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/worldgen)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/worldgen/terrain)