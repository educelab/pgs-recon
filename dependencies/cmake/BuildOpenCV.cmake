option(BUILD_OPENCV "Build OpenCV" ON)
if(BUILD_OPENCV)
externalproject_add(
    opencv
    DEPENDS jpeg
    URL https://github.com/opencv/opencv/archive/3.4.15.tar.gz
    URL_HASH SHA512=775149e56f0aa94d53eb024404866380d97ce423ef1c8343ee8f12c1377e454ae182b2528e86949b5f7250e551d464bd1a5de2e2d9f0d0e1dd3dc188a1db790d
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
