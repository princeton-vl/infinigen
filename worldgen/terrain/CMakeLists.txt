cmake_minimum_required(VERSION 3.2)
project(terrain)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_compile_definitions(M_PI 3.14159265358979323846)
add_compile_definitions(M_PI_2 1.57079632679489661923)


find_package(OpenMP REQUIRED)

set(DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)


set(elements
    "voronoi_rocks"
    "upsidedown_mountains"
    "ground"
    "warped_rocks"
    "mountains"
    "landtiles"
    "atmosphere"
    "waterbody"
)

set(surfaces
    "chunkyrock"
    "cobble_stone"
    "cracked_ground"
    "dirt"
    "ice"
    "mountain"
    "mud"
    "sand"
    "sandstone"
    "snow"
    "soil"
    "stone"
)


# cuda part
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler")
    cuda_add_library(FastNoiseLite SHARED source/cuda/utils/FastNoiseLite.cu)
    install(TARGETS FastNoiseLite DESTINATION ${DESTINATION}/cuda/utils)
    foreach(element ${elements})
        cuda_add_library(${element} SHARED source/cuda/elements/${element}.cu)
        install(TARGETS ${element} DESTINATION ${DESTINATION}/cuda/elements)
    endforeach()
    foreach(surface ${surfaces})
        cuda_add_library(${surface} SHARED source/cuda/surfaces/${surface}.cu)
        install(TARGETS ${surface} DESTINATION ${DESTINATION}/cuda/surfaces)
    endforeach()
endif()

# cpu part
if(WIN32)
    foreach(element ${elements})
        add_library(${element} SHARED source/cpu/elements/${element}.cpp)
        target_link_libraries(${element} PRIVATE OpenMP::OpenMP_CXX)
        install(TARGETS ${element} DESTINATION ${DESTINATION}/cpu/elements)
    endforeach()
    foreach(surface ${surfaces})
        add_library(${surface} SHARED source/cpu/surfaces/${surface}.cpp)
        target_link_libraries(${surface} PRIVATE OpenMP::OpenMP_CXX)
        install(TARGETS ${surface} DESTINATION ${DESTINATION}/cpu/surfaces)
    endforeach()
    add_library(FastNoiseLite SHARED source/cpu/utils/FastNoiseLite.cpp)
    target_link_libraries(FastNoiseLite PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS FastNoiseLite DESTINATION ${DESTINATION}/cpu/utils)
endif()

file(GLOB_RECURSE MESHING_SOURCES source/cpu/meshing/*.cpp)
foreach(meshing_source ${MESHING_SOURCES})
    get_filename_component(meshing_name ${meshing_source} NAME_WE)
    add_library(${meshing_name} SHARED ${meshing_source})
    target_link_libraries(${meshing_name} PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS ${meshing_name} DESTINATION ${DESTINATION}/cpu/meshing)
endforeach()

add_library(SoilMachine SHARED source/cpu/soil_machine/SoilMachine.cpp)
target_include_directories(SoilMachine PRIVATE ${CONDA_INDICATOR_DIRECTORY})
target_link_libraries(SoilMachine PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS SoilMachine DESTINATION ${DESTINATION}/cpu/SoilMachine)

add_custom_target(terrin_python_ext
    COMMAND ${Python_EXECUTABLE} setup.py build_ext --inplace
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building python extension"
)