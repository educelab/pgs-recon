# openMVG statically links a bundled zlib whenever it can't find libpng or
# libtiff. Mixed with a system libpng/libtiff, that zlib wins at symbol
# resolution and image reads abort at runtime, so require the two to agree:
# both found, or neither.
find_package(PNG QUIET)
find_package(TIFF QUIET)
set(_mvg_have "")
if(PNG_FOUND AND NOT TIFF_FOUND)
  set(_mvg_have "libpng")
  set(_mvg_need "libtiff (libtiff-dev)")
elseif(TIFF_FOUND AND NOT PNG_FOUND)
  set(_mvg_have "libtiff")
  set(_mvg_need "libpng (libpng-dev)")
endif()
if(NOT _mvg_have STREQUAL "")
  message(FATAL_ERROR
      "openMVG found ${_mvg_have} but not ${_mvg_need}, which builds binaries "
      "that abort when reading images. Install ${_mvg_need}, or remove the "
      "${_mvg_have} development headers to use openMVG's bundled copies of both.")
endif()
unset(_mvg_have)
unset(_mvg_need)

ExternalProject_Add(
    openMVG
    DEPENDS eigen ceres jpeg
    GIT_REPOSITORY https://github.com/openMVG/openMVG.git
    GIT_TAG c92ed1b
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    SOURCE_SUBDIR src
    PATCH_COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVG-v1.6-UpdateSensorDB.diff || true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DEIGENSPARSE:BOOL=ON
        -DOpenMVG_BUILD_DOC:BOOL=OFF
        -DOpenMVG_BUILD_EXAMPLES:BOOL=OFF
        -DOpenMVG_BUILD_GUI_SOFTWARES:BOOL=OFF
)