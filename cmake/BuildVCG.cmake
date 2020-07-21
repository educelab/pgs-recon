# Note: Must use the cdcseacave repo
ExternalProject_Add(
    VCG
    URL https://github.com/cdcseacave/VCG/archive/18be260.tar.gz
    URL_HASH SHA256=c7e5f9d2aebef4516f172f78cc5f0787f23b2b366681f4927a0b61822570aa30
    DOWNLOAD_NO_PROGRESS true
    BUILD_IN_SOURCE true
    CONFIGURE_COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/vcg
    BUILD_COMMAND ""
    INSTALL_COMMAND cp -R . ${CMAKE_INSTALL_PREFIX}/include/vcg
)
