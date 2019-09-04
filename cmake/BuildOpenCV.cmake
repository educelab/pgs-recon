option(BUILD_OPENCV "Build OpenCV" ON)
if(BUILD_OPENCV)
externalproject_add(
    opencv
    DEPENDS jpeg
    URL https://github.com/opencv/opencv/archive/3.4.6.tar.gz
    URL_HASH SHA1=c445fe9f0a07a95837d2f39c0637da71158129c5
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
