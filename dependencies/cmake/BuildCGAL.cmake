option(BUILD_CGAL "Build CGAL" ON)
if(BUILD_CGAL)
  ExternalProject_Add(
      CGAL
      DEPENDS eigen
      URL https://github.com/CGAL/cgal/releases/download/v6.0.3/CGAL-6.0.3-library.tar.xz
      URL_HASH SHA512=a2ebf1fbd9a801f75296f1fec13b8325c7485ed690a458fdd017c7893064a2d1980234251455b52a827c217d84b508f831e433e2698dab3fe446d9f46fc8bf10
      DOWNLOAD_NO_PROGRESS ON
      DOWNLOAD_EXTRACT_TIMESTAMP OFF
      CMAKE_CACHE_ARGS
          ${GLOBAL_CMAKE_ARGS}
          -DCMAKE_BUILD_TYPE:STRING=Release
          -DWITH_CGAL_Qt5:BOOL=OFF
          -DWITH_examples:BOOL=OFF
          -DWITH_demos:BOOL=OFF
  )
else()
  find_package(CGAL CONFIG REQUIRED)
  add_custom_target(CGAL)
endif()
