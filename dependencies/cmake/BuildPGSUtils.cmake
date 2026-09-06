# Note: VCG_ROOT is set the way BuildOpenMVS.cmake sets it -- the install prefix
# first -- because pgs-decimate is built against the same headers OpenMVS is.
ExternalProject_Add(
    pgs-recon-utilities
    DEPENDS eigen opencv openMVG VCG
    URL ${PROJECT_SOURCE_DIR}/utilities
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DVCG_ROOT:PATH=${CMAKE_INSTALL_PREFIX}/include/vcg
)