# Note: Must use the cdcseacave repo
ExternalProject_Add(
    VCG
    URL https://github.com/cdcseacave/VCG/archive/56c76b0.tar.gz
    URL_HASH SHA256=770c2b6c9af19fb71c9db2b3a26627037f3393b7d09d8ba601616e346ecf2bf0
    DOWNLOAD_NO_PROGRESS true
    BUILD_IN_SOURCE true
    CONFIGURE_COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/vcg
    BUILD_COMMAND ""
    INSTALL_COMMAND cp -R . ${CMAKE_INSTALL_PREFIX}/include/vcg
)
