option(BUILD_CGAL "Build CGAL" ON)
if(BUILD_CGAL)
  ExternalProject_Add(
      CGAL
      DEPENDS eigen
      URL https://github.com/CGAL/cgal/releases/download/v5.6.2/CGAL-5.6.2-library.tar.xz
      URL_HASH SHA512=d81f918d3f7f48f1efe4e5e1fcedd5a91ffb61c25b0aca10ec5b17303f6bd86eef2eae9cac1d2c0520ec18fc1b3459a49427e4b58cfa9db5a5e0cd5ab2f618b9
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
