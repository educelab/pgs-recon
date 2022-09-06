ExternalProject_Add(
    openMVG
    DEPENDS eigen jpeg opencv
    GIT_REPOSITORY https://github.com/openMVG/openMVG
    GIT_TAG 97fd1e2
    DOWNLOAD_NO_PROGRESS true
    SOURCE_SUBDIR src
    PATCH_COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVG-v1.6-UpdateSensorDB.diff || true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DEIGENSPARSE:BOOL=ON
        -DOpenMVG_USE_OCVSIFT:BOOL=ON
        -DOpenMVG_BUILD_DOC:BOOL=OFF
        -DOpenMVG_BUILD_EXAMPLES:BOOL=OFF
        -DOpenMVG_BUILD_GUI_SOFTWARES:BOOL=OFF
)