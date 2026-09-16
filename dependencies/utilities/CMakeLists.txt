cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(pgs-recon-utils VERSION 1.0.0)

# find dependencies
find_package(OpenMVG REQUIRED)
find_package(OpenCV 4.7 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS program_options)

# VCG: header-only, no package config, so cmake/FindVCG.cmake finds it via -DVCG_ROOT.
# Appended here, not at the top: find_package(OpenMVG) drops what the caller appended.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(VCG REQUIRED)

# header-only dependencies
include(FetchContent)
# rc.3 for two PLY changes pgs-decimate needs: reading OpenMVS's sized type
# aliases (educelab/libcore#24) and writing binary PLY (#25, #28).
FetchContent_Declare(
    libcore
    GIT_REPOSITORY https://github.com/educelab/libcore.git
    GIT_TAG v0.3.0-rc.3
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(libcore)
# bvh (madmann91 v2): header-only ray/triangle BVH, the renderer's nearest-hit
# structure. Same pin registration-toolkit uses, so pgs-localize's render and
# rt_reorder_texture's stay comparable -- which is what keeps the measured
# numbers in ADR 0011 honest.
FetchContent_Declare(
    bvh
    GIT_REPOSITORY https://github.com/madmann91/bvh.git
    GIT_TAG 66e445b
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(bvh)
FetchContent_Declare(
    indicators
    GIT_REPOSITORY https://github.com/p-ranav/indicators.git
    GIT_TAG 3872f37
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(indicators)

# shared marker detection/triangulation library (used by global_scaler and sfm_orient)
add_library(marker_detection STATIC
        src/RANSAC.hpp
        src/marker_detection.hpp
        src/marker_detection.cpp
        src/marker_types.hpp)
target_include_directories(marker_detection
    PUBLIC
        src/
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${indicators_SOURCE_DIR}/include>
)
target_link_libraries(marker_detection
    Eigen3::Eigen
    opencv_calib3d
    opencv_core
    opencv_imgproc
    opencv_objdetect
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_geometry
    OpenMVG::openMVG_multiview
    OpenMVG::openMVG_sfm
)
target_compile_features(marker_detection PUBLIC cxx_std_17)

# global scaler utility
add_executable(global_scaler
        src/global_scaler.cpp
        src/global_scaler_io.hpp
        src/global_scaler_io.cpp)
target_include_directories(global_scaler
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${indicators_SOURCE_DIR}/include>
)
target_link_libraries(global_scaler
    marker_detection
    Boost::program_options
    Eigen3::Eigen
    opencv_core
    opencv_imgcodecs
    opencv_objdetect
    OpenMVG::openMVG_geometry
    OpenMVG::openMVG_sfm
)
target_compile_features(global_scaler PUBLIC cxx_std_17)
set_target_properties(global_scaler PROPERTIES OUTPUT_NAME "pgs-global-scaler")

# sfm orient utility
add_executable(sfm_orient src/sfm_orient.cpp src/Npy.hpp)
target_include_directories(sfm_orient
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${indicators_SOURCE_DIR}/include>
)
target_link_libraries(sfm_orient
    marker_detection
    Boost::program_options
    Eigen3::Eigen
    opencv_core
    opencv_imgcodecs
    opencv_objdetect
    OpenMVG::openMVG_geometry
    OpenMVG::openMVG_sfm
)
target_compile_features(sfm_orient PUBLIC cxx_std_17)
set_target_properties(sfm_orient PROPERTIES OUTPUT_NAME "pgs-sfm-orient")

# error-bounded decimation utility
add_executable(decimate
        src/decimate.cpp
        src/decimate_core.hpp
        src/decimate_core.cpp)
target_include_directories(decimate
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${indicators_SOURCE_DIR}/include>
)
target_link_libraries(decimate
    Boost::program_options
    Eigen3::Eigen
    VCG::VCG
)
target_compile_features(decimate PUBLIC cxx_std_17)
set_target_properties(decimate PROPERTIES OUTPUT_NAME "pgs-decimate")

# generate aruco markers utility
add_executable(gen_aruco_markers src/gen_aruco_markers.cpp)
target_include_directories(gen_aruco_markers
    PUBLIC
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
)
target_link_libraries(gen_aruco_markers
        Boost::program_options
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_objdetect
)
target_compile_features(gen_aruco_markers PUBLIC cxx_std_17)
set_target_properties(gen_aruco_markers PROPERTIES OUTPUT_NAME "pgs-generate-markers")

# The describer reader, alone in a C++17 library: openMVG's bundled cereal does
# not compile under C++20 (its StaticObject::LockGuard declares a defaulted copy
# constructor, which P1008 stopped leaving an aggregate, so cereal's own
# `return LockGuard{};` finds no constructor). pgs-localize is C++20 because bvh
# needs std::span, so the one file that has to meet cereal is held back rather
# than the whole tool.
add_library(localize_describer STATIC
        src/localize_describer.hpp
        src/localize_describer.cpp)
# openMVG's nonFree tree (the VLFeat SIFT describer) is installed but no
# imported target carries its include path. Derive it from the one target that
# points into openMVG_dependencies rather than guessing at the prefix.
get_target_property(_mvg_cereal_include OpenMVG::cereal INTERFACE_INCLUDE_DIRECTORIES)
get_filename_component(_mvg_deps_include "${_mvg_cereal_include}" DIRECTORY)
get_filename_component(_mvg_deps_include "${_mvg_deps_include}" DIRECTORY)
if(NOT EXISTS "${_mvg_deps_include}/nonFree/sift/SIFT_describer_io.hpp")
    message(FATAL_ERROR
        "openMVG's nonFree SIFT describer headers were not found under "
        "${_mvg_deps_include}. pgs-localize needs them to read the describer "
        "recorded in a matches directory's image_describer.json.")
endif()
target_include_directories(localize_describer
    PUBLIC
        src/
        "${_mvg_deps_include}"
)
target_link_libraries(localize_describer
    Eigen3::Eigen
    OpenMVG::cereal
    OpenMVG::openMVG_features
    OpenMVG::vlsift
)
target_compile_features(localize_describer PUBLIC cxx_std_17)

# localization utility
add_executable(localize
        src/localize.cpp
        src/localize_camera.hpp
        src/localize_camera.cpp
        src/localize_match.hpp
        src/localize_match.cpp
        src/localize_render.hpp
        src/localize_render.cpp
        src/localize_solve.hpp
        src/localize_solve.cpp
        src/localize_sparse.hpp
        src/localize_sparse.cpp
        src/localize_describer.hpp
        src/Npy.hpp)
target_include_directories(localize
    PUBLIC
        src/
        $<BUILD_INTERFACE:${libcore_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${bvh_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${indicators_SOURCE_DIR}/include>
)
target_link_libraries(localize
    localize_describer
    Boost::program_options
    Eigen3::Eigen
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_video
    OpenMVG::cereal
    OpenMVG::openMVG_camera
    OpenMVG::openMVG_features
    OpenMVG::openMVG_geometry
    OpenMVG::openMVG_matching
    OpenMVG::openMVG_multiview
    OpenMVG::openMVG_sfm
)
# C++20, alone among the utilities: bvh v2 needs it (std::span, defaulted
# comparison operators), and registration-toolkit's rt_core builds at the same
# standard against the same pin.
target_compile_features(localize PUBLIC cxx_std_20)
set_target_properties(localize PROPERTIES OUTPUT_NAME "pgs-localize")

# install
include(GNUInstallDirs)
install(TARGETS
    global_scaler
    localize
    sfm_orient
    gen_aruco_markers
    decimate
)
