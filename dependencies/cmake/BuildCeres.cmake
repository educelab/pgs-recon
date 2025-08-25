option(BUILD_CERES "Build Ceres" OFF)
if(BUILD_CERES)
  ExternalProject_Add(
      ceres
      DEPENDS eigen
      URL http://ceres-solver.org/ceres-solver-2.2.0.tar.gz
      URL_HASH SHA512=bf519fbcbd0ee2d4624be72cde061a09d191ee5e56dc33984669393799b885c2164efc99dcef71307f8b50a9ccad945882376ace3d9a44ba7e1f25c20511aabd
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          ${GLOBAL_CUDA_ARGS}
          -DUSE_CUDA:BOOL=${USE_CUDA}
          -DMINIGLOG:BOOL=ON
  )
else()
  # OpenMVG builds its own copy if we don't build it here
  add_custom_target(ceres)
endif()