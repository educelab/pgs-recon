option(BUILD_EIGEN "Build Eigen" ON)
if(BUILD_EIGEN)
  ExternalProject_Add(
      eigen
      URL https://github.com/eigenteam/eigen-git-mirror/archive/3.2.10.tar.gz
      URL_HASH SHA256=ba4f95f9082ce99012d5ff320472a0e9fd0be921a20f11912f0bf7e6ca93f240
      DOWNLOAD_NO_PROGRESS true
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
  )
else()
  find_package(Eigen3 3.2 NO_MODULE EXACT REQUIRED)
  add_custom_target(eigen)
endif()
