cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

#### Initialize cache variables ####
get_property(multiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT multiConfig AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

#### Define project ####
project(Photogrammetry)

#### CMake Module Path ####
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

### Include extra CMake Modules ###
include(ExternalProject)
include(CMakeDependentOption)

#### Setup output directory if not specified ####
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/installed" CACHE PATH "Default Install Path" FORCE )
endif()

### Global arguments ###
set(GLOBAL_CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
    "-DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}"
    "-DBUILD_SHARED_LIBS:BOOL=FALSE"
)

### Eigen ###
include(BuildEigen)

### jpeg ###
include(BuildJPEG)

### OpenCV ###
include(BuildOpenCV)

### CGAL ###
include(BuildCGAL)

### VCG ###
include(BuildVCG)

### openMVG ###
include(BuildOpenMVG)

### openMVS ###
include(BuildOpenMVS)
