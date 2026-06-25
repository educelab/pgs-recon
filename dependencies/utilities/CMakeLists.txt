cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(pgs-recon-utils VERSION 1.0.0)

# find dependencies
find_package(OpenMVG REQUIRED)
find_package(OpenCV 4.7 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Boost 1.70 CONFIG REQUIRED COMPONENTS program_options)

# header-only dependencies
include(FetchContent)
FetchContent_Declare(
    libcore
    GIT_REPOSITORY https://github.com/educelab/libcore.git
    GIT_TAG 8eb070c979938131382d6faa1f4c91ccc0ad1aa9
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(libcore)
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
add_executable(sfm_orient src/sfm_orient.cpp)
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

# install
include(GNUInstallDirs)
install(TARGETS
    global_scaler
    sfm_orient
    gen_aruco_markers
)