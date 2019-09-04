option(BUILD_CGAL "Build CGAL" ON)
if(BUILD_CGAL)
  ExternalProject_Add(
      CGAL
      DEPENDS eigen
      URL https://github.com/CGAL/cgal/archive/84b9328.tar.gz
      URL_HASH SHA256=313dd7dbba9a67ecdef4d82265bede4cd9a3d8409bdc647f472197301b61db01
      DOWNLOAD_NO_PROGRESS true
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DCMAKE_BUILD_TYPE:STRING=Release
  )
else()
  find_package(CGAL CONFIG REQUIRED)
  add_custom_target(CGAL)
endif()
