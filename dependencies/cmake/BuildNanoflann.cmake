# nanoflann: header-only KD-tree library. Required by OpenMVS >= v2.4.0
# (libs/Common). Installs a config file to <prefix>/share/cmake/nanoflann.
option(BUILD_NANOFLANN "Build nanoflann" ON)
if(BUILD_NANOFLANN)
  ExternalProject_Add(
      nanoflann
      URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/1.10.1.tar.gz
      URL_HASH SHA512=95c4e929c9e5accc99ac5f2927119ed3f8240729ed7695ed1fbe89ee96ba3498c1aa25164c57819d44f0f829a7c2b46c468932bb0bfb665bb01f60bf1bb9e02a
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DNANOFLANN_BUILD_EXAMPLES:BOOL=OFF
          -DNANOFLANN_BUILD_TESTS:BOOL=OFF
  )
else()
  find_package(nanoflann CONFIG REQUIRED)
  add_custom_target(nanoflann)
endif()
