option(BUILD_EIGEN "Build Eigen" ON)
if(BUILD_EIGEN)
  ExternalProject_Add(
      eigen
      URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
      URL_HASH SHA256=d56fbad95abf993f8af608484729e3d87ef611dd85b3380a8bad1d5cbc373a57
      DOWNLOAD_NO_PROGRESS true
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
  )
else()
  find_package(Eigen3 3.3 REQUIRED)
  add_custom_target(eigen)
endif()
