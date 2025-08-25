# Note: Setting VCG_ROOT forces looking in the install prefix *first*
ExternalProject_Add(
    openMVS
    DEPENDS eigen VCG CGAL jpeg opencv
    GIT_REPOSITORY https://github.com/cdcseacave/openMVS.git
    GIT_TAG 36ffb22
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    PATCH_COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.3-FixCPP20.diff || true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        ${GLOBAL_CUDA_ARGS}
        -DVCG_ROOT:PATH=${CMAKE_INSTALL_PREFIX}/include/vcg
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DOpenMVS_USE_PYTHON:BOOL=OFF
        -DOpenMVS_USE_CUDA:BOOL=${USE_CUDA}
        -DOpenMVS_USE_SSE:BOOL=${HAS_SSE}
        -DENABLE_PRECOMPILED_HEADERS:BOOL=OFF
)
ExternalProject_Get_Property(openMVS SOURCE_DIR)
ExternalProject_Add_Step(
    openMVS
    extra_patch
    COMMENT "Performing patch step #2 for 'openMVS'"
    COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.3-FixFindBoost.diff || true
    WORKING_DIRECTORY ${SOURCE_DIR}
    DEPENDEES patch
    DEPENDERS configure
)