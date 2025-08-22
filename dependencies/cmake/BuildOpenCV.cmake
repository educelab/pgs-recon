option(BUILD_OPENCV "Build OpenCV" ON)
if(BUILD_OPENCV)
externalproject_add(
    opencv
    DEPENDS jpeg
    URL https://github.com/opencv/opencv/archive/refs/tags/4.12.0.tar.gz
    URL_HASH SHA512=8ac63ddd61e22cc0eaeafee4f30ae6e1cab05fc4929e2cea29070203b9ca8dfead12cc0fd7c4a87b65c1e20ec6b9ab4865a1b83fad33d114fc0708fdf107c51b
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DWITH_VTK:BOOL=OFF
        -DBUILD_JPEG:BOOL=OFF
        -DBUILD_TESTS:BOOL=OFF
        -DBUILD_PERF_TESTS:BOOL=OFF
        -DBUILD_EXAMPLES:BOOL=OFF
        -DWITH_FFMPEG:BOOL=OFF
        -DWITH_OPENEXR:BOOL=OFF
        -DOBSENSOR_USE_ORBBEC_SDK:BOOL=OFF
        -DWITH_OBSENSOR:BOOL=OFF
)
else()
  find_package(OpenCV CONFIG REQUIRED)
  add_custom_target(opencv)
endif()
