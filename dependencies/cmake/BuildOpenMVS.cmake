# Note: Setting VCG_ROOT forces looking in the install prefix *first*
ExternalProject_Add(
    openMVS
    DEPENDS eigen VCG CGAL jpeg opencv nanoflann
    GIT_REPOSITORY https://github.com/cdcseacave/openMVS.git
    # Pinned to a post-v2.4.0 develop commit (2026-03-20). The v2.4.0 release
    # (pgs-recon MR !58) shipped a bug; this commit picks up the upstream fix
    # while still self-reporting version 2.4.0. All patches below still apply.
    # This commit also introduced a CUDA-only compile bug (undefined xMax/yMax
    # in SceneRefineCUDA.cu); FixRefineCUDABounds below cherry-picks the upstream
    # fix (develop a4d652a) rather than bumping the pin, since later commits pull
    # in new SfM dependencies. Only USE_CUDA=ON builds hit this, so CI misses it.
    GIT_TAG ca991d50964ad2cfabc94c88d61a444999670a5d
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    PATCH_COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.4-FixFindBoost.diff || true
      COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.4-OptionalJXL.diff || true
      COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.4-RemoveMTLTransparency.diff || true
      COMMAND patch -p1 --forward -i ${CMAKE_SOURCE_DIR}/patches/openMVS-v2.4-FixRefineCUDABounds.diff || true
    CMAKE_CACHE_ARGS
        ${GLOBAL_CMAKE_ARGS}
        ${GLOBAL_CUDA_ARGS}
        -DVCG_ROOT:PATH=${CMAKE_INSTALL_PREFIX}/include/vcg
        -DEigen3_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/eigen3/cmake/
        -DOpenMVS_USE_PYTHON:BOOL=OFF
        -DOpenMVS_USE_CUDA:BOOL=${USE_CUDA}
        -DOpenMVS_USE_SSE:BOOL=${HAS_SSE}
        -DOpenMVS_BUILD_VIEWER:BOOL=OFF
        -DENABLE_PRECOMPILED_HEADERS:BOOL=OFF
)
