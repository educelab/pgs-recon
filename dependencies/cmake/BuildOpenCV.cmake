# Note: OpenMVS requires cv::String which is removed in OpenCV 4+
option(BUILD_OPENCV "Build OpenCV" ON)
if(BUILD_OPENCV)
externalproject_add(
    opencv
    DEPENDS jpeg
    URL https://github.com/opencv/opencv/archive/refs/tags/3.4.16.tar.gz
    URL_HASH SHA512=2fa9243625309a1c12c916737c94f0e2f9566f0828469b148cc1683dd1b8db8d1d58f90a36cfcaa72052964a718929451f04cda5361d8b546a63da69217d040a
    DOWNLOAD_NO_PROGRESS true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DBUILD_JPEG:BOOL=OFF
        -DBUILD_TESTS:BOOL=OFF
        -DWITH_FFMPEG:BOOL=OFF
)
else()
  find_package(OpenCV CONFIG REQUIRED)
  add_custom_target(opencv)
endif()
