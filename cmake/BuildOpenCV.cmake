option(BUILD_OPENCV "Build OpenCV" ON)
if(BUILD_OPENCV)
externalproject_add(
    opencv
    DEPENDS jpeg
    URL https://github.com/opencv/opencv/archive/3.4.10.tar.gz
    URL_HASH SHA256=1ed6f5b02a7baf14daca04817566e7c98ec668cec381e0edf534fa49f10f58a2
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
