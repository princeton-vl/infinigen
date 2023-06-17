cmake_minimum_required(VERSION 3.10.0)
project(process_mesh VERSION 0.1.0)
add_definitions(-D GLM_ENABLE_EXPERIMENTAL)
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

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL GLX)

file(GLOB ADJACENT_SRC_FILES
     "*.hpp"
     "*.cpp"
)

# (Only needed on my desktop). This line was taken from https://stackoverflow.com/a/52137618/5057543
link_libraries(stdc++fs)

add_executable(process_mesh ${ADJACENT_SRC_FILES} dependencies/scharstein_flow_viz/colorcode.cpp dependencies/fast_obj/fast_obj.h dependencies/fast_obj/fast_obj.c dependencies/glad/src/glad.c dependencies/glad/src/glad_egl.c)

set_property(TARGET process_mesh PROPERTY CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

target_link_libraries(process_mesh GL GLU dl EGL m stdc++ X11 cnpy)
