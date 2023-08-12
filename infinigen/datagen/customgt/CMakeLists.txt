cmake_minimum_required(VERSION 3.10.0)
project(customgt VERSION 0.1.0)

if (NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin" OR ${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not supported")
endif()

add_definitions(-D GLM_ENABLE_EXPERIMENTAL)
add_definitions(-D PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR})
include_directories(/usr/include/) # on linux
include_directories(dependencies) # for cnpy
include_directories(dependencies/indicators/include)
include_directories(dependencies/argparse/include)
include_directories(dependencies/eigen)
include_directories(dependencies/tinycolormap/include)
include_directories(dependencies/glm)
include_directories(dependencies/json/single_include)
include_directories(dependencies/glad/include)
include_directories(dependencies/scharstein_flow_viz)

add_subdirectory(dependencies/cnpy)
add_subdirectory(dependencies/fast_obj)
add_subdirectory(dependencies/tinycolormap)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-D SYSTEM_NUM=1)
    include_directories(dependencies/glfw/include)
    add_subdirectory(dependencies/glfw)
    find_package(OpenGL REQUIRED)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    add_definitions(-D SYSTEM_NUM=0)
    find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)
endif()

file(GLOB ADJACENT_SRC_FILES
     "*.hpp"
     "*.cpp"
)

# This line was taken from https://stackoverflow.com/a/52137618/5057543
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    link_libraries(stdc++fs)
endif()

add_executable(customgt ${ADJACENT_SRC_FILES} dependencies/scharstein_flow_viz/colorcode.cpp dependencies/fast_obj/fast_obj.h dependencies/fast_obj/fast_obj.c dependencies/glad/src/glad.c dependencies/glad/src/glad_egl.c)

set_property(TARGET customgt PROPERTY CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_libraries(customgt dl m stdc++ cnpy glfw)
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(customgt GL GLU dl EGL m stdc++ X11 cnpy)
endif()
