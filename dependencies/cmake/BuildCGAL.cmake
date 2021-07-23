option(BUILD_CGAL "Build CGAL" ON)
if(BUILD_CGAL)
  ExternalProject_Add(
      CGAL
      DEPENDS eigen
      URL https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-5.0.2/CGAL-5.0.2-library.tar.xz
      URL_HASH SHA256=bdc029e87902f76ad6b55af59cf595c3a974f3c8eac6e4a6df3c8eeebfab5d0c
      DOWNLOAD_NO_PROGRESS true
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
