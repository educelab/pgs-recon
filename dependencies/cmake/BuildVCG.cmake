# Note: Must use the cdcseacave repo
ExternalProject_Add(
    VCG
    URL https://github.com/cdcseacave/VCG/archive/88f12f2.tar.gz
    URL_HASH SHA512=170b3bf61a58802ea9a593c0b9356f2db92a878140570ba169adf7f62c86e542a6170cc1601095514e15f409cd3a0798879941e5909b739e549452171b9d4eae
    DOWNLOAD_NO_PROGRESS true
    BUILD_IN_SOURCE true
    CONFIGURE_COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/vcg
    BUILD_COMMAND ""
    INSTALL_COMMAND cp -R . ${CMAKE_INSTALL_PREFIX}/include/vcg
)
