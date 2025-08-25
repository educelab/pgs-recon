# The recommended way to install is to copy the files to ${PREFIX}/include/
ExternalProject_Add(
    VCG
    URL https://github.com/cnr-isti-vclab/vcglib/archive/refs/tags/2025.07.tar.gz
    URL_HASH SHA512=6533dfdc48a8ee0f904c49edcd25a3c06a945cec7baa047ddbba78ae48fbf7b490718fe15eb7c729f9c097114b798ec5204302b37011906a0bed4de819616717
    DOWNLOAD_NO_PROGRESS ON
    DOWNLOAD_EXTRACT_TIMESTAMP OFF
    BUILD_IN_SOURCE ON
    CONFIGURE_COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/vcg
    BUILD_COMMAND ""
    INSTALL_COMMAND cp -R . ${CMAKE_INSTALL_PREFIX}/include/vcg
)
