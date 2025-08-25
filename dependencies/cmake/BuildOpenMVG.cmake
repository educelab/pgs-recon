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