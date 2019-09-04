# Note: Setting VCG_ROOT forces looking in the install prefix *first*
ExternalProject_Add(
    openMVS
    DEPENDS eigen VCG CGAL jpeg opencv
    URL https://github.com/cdcseacave/openMVS/archive/afe30ae.tar.gz
    URL_HASH SHA256=5f03414908fbb9ebc79ea4634043feebf380465ad3dd5e299fbfbd3d60ae3a9b
    DOWNLOAD_NO_PROGRESS true
    PATCH_COMMAND patch -p1 -i ${CMAKE_SOURCE_DIR}/patches/openMVS-afe30ae-FindCGAL.diff
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DVCG_ROOT:PATH=${CMAKE_INSTALL_PREFIX}/include/vcg
)
