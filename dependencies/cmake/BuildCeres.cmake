option(BUILD_CERES "Build Ceres" OFF)
if(BUILD_CERES)
  ExternalProject_Add(
      ceres
      DEPENDS eigen
      URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz
      URL_HASH SHA512=d4cefe5851e25bd3c7b76352092d8d549eb371af2e35a325736554c54fe58a3523658697c3e2d65af660fe6895ae3d96fe31bd1875870474fc4b6fed3bbdfae9
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