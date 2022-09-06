option(BUILD_CGAL "Build CGAL" ON)
if(BUILD_CGAL)
  ExternalProject_Add(
      CGAL
      DEPENDS eigen
      URL https://github.com/CGAL/cgal/releases/download/v5.5/CGAL-5.5-library.tar.xz
      URL_HASH SHA512=0ec4760c7d7306077452dac3c4c725d3c75bd72c278049df367aa42972ad81cc22aa4af12eca6a23940eceded54cdd88a092e9a007933ad7d8dc7aa17e8ed6fe
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
