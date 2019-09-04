ExternalProject_Add(
    openMVG
    DEPENDS eigen jpeg opencv
    GIT_REPOSITORY https://github.com/openMVG/openMVG
    GIT_TAG 835f1e585377a55d4143548fa9181438f5d23889
    DOWNLOAD_NO_PROGRESS true
    SOURCE_SUBDIR src
    PATCH_COMMAND patch -p1 -i ${CMAKE_SOURCE_DIR}/patches/openMVG-835f1e5-UpdateSensorDB.diff
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        -DEIGENSPARSE:BOOL=ON
        -OpenMVG_USE_OPENCV:BOOL=ON
        -OpenMVG_USE_OCVSIFT:BOOL=ON
)
