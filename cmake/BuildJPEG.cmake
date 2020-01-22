option(BUILD_JPEG "Build JPEG" ON)
cmake_dependent_option(BUILD_JPEG_TURBO "Build jpeg-turbo" OFF "BUILD_JPEG" OFF)
if(BUILD_JPEG)
  # Pick which libjpeg to build
  if(BUILD_JPEG_TURBO)
    # jpeg-turbo (Default)
    find_package(NASM REQUIRED)
    ExternalProject_Add(
      jpeg
      URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.2.tar.gz
      URL_HASH SHA256=b45255bd476c19c7c6b198c07c0487e8b8536373b82f2b38346b32b4fa7bb942
      DOWNLOAD_NO_PROGRESS true
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DENABLE_SHARED:BOOL=FALSE
    )

  else()
    # libjpeg
    ExternalProject_Add(
      jpeg
      URL https://github.com/csparker247/jpeg-cmake/archive/v1.1.0.tar.gz
      URL_HASH SHA256=e4d91447b8718df8b0dc7b6e01be3cc27e6b2f90d40450ea4c760eda6b843548
      DOWNLOAD_NO_PROGRESS true
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DBUILD_SHARED_LIBS:BOOL=FALSE
    )
  endif(BUILD_JPEG_TURBO)

else()
  find_package(JPEG REQUIRED)
  add_custom_target(jpeg)
endif()
