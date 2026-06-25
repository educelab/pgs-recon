option(BUILD_JPEG "Build JPEG" ON)
cmake_dependent_option(BUILD_JPEG_TURBO "Build jpeg-turbo" OFF "BUILD_JPEG" OFF)
if(BUILD_JPEG)
  # Pick which libjpeg to build
  if(BUILD_JPEG_TURBO)
    # jpeg-turbo (Default)
    find_package(NASM REQUIRED)
    ExternalProject_Add(
      jpeg
      URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
      URL_HASH SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DENABLE_SHARED:BOOL=FALSE
    )

  else()
    # libjpeg
    ExternalProject_Add(
      jpeg
      URL https://github.com/csparker247/jpeg-cmake/archive/v10.0.0.tar.gz
      URL_HASH SHA512=b6947e758d36ff97bd07a818a72e5c8c2d53b9d63978618834391e482a421adf7b57dfb1e0e868b9205b1bab960ad72001898a7d5424ba2ecc18c8f2a89c7a1d
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DBUILD_SHARED_LIBS:BOOL=FALSE
    )
  endif(BUILD_JPEG_TURBO)

else()
  find_package(JPEG REQUIRED)
  add_custom_target(jpeg)
endif()
