#[=======================================================================[.rst:
FindVCG
-------

Finds vcglib, the header-only mesh processing library.

No tagged vcglib release ships CMake install or export rules -- 2025.07, the
newest, has neither, which is why the superbuild installs it by copying the
source tree (``BuildVCG.cmake``) and why this module exists. Upstream ``devel``
gained ``install(EXPORT vcglibTargets)`` and a ``vcglibConfig.cmake`` in July
2026; once a release carries it, this module can be dropped in favor of
``find_package(vcglib CONFIG)`` and the ``vcglib::vcglib`` target it defines.

Imported target
^^^^^^^^^^^^^^^

``VCG::VCG``
  The header-only library, if found.

Result variables
^^^^^^^^^^^^^^^^

``VCG_FOUND``, ``VCG_INCLUDE_DIRS``

Hints
^^^^^

``VCG_ROOT`` (variable or environment) is searched first, so a copy in an
install prefix wins over a system one.

Layouts
^^^^^^^

``vcg/complex/complex.h`` includes ``<wrap/callback.h>`` unconditionally, and
where that file lands differs by how vcglib was installed: a sibling of ``vcg/``
in the copied source tree, but ``include/vcg/wrap/callback.h`` under one of
upstream's install rules. So the directory that resolves it is searched for
separately, and both go on the target. Three layouts are known to work: the
superbuild's copied source tree (``<root>/vcg`` + ``<root>/wrap``), an upstream
``make install`` (``include/vcg`` + ``include/wrap``), and headers-only
packaging (``include/vcg`` + ``include/vcg/wrap``).
#]=======================================================================]

find_path(VCG_INCLUDE_DIR
    NAMES
        "vcg/complex/complex.h"
    HINTS
        "${VCG_ROOT}"
        "$ENV{VCG_ROOT}"
    PATH_SUFFIXES
        "vcg"
        "include"
        "include/vcg"
    DOC
        "Directory holding vcg/complex/complex.h")

# The directory <wrap/callback.h> resolves against: alongside vcg/ in a source
# tree, one level down in an installed one.
find_path(VCG_WRAP_INCLUDE_DIR
    NAMES
        "wrap/callback.h"
    HINTS
        "${VCG_INCLUDE_DIR}"
        "${VCG_INCLUDE_DIR}/vcg"
    NO_DEFAULT_PATH
    DOC
        "Directory holding wrap/callback.h")
mark_as_advanced(VCG_INCLUDE_DIR VCG_WRAP_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(VCG
    REQUIRED_VARS
        VCG_INCLUDE_DIR
        VCG_WRAP_INCLUDE_DIR
    REASON_FAILURE_MESSAGE
        "set VCG_ROOT to the directory holding vcg/complex/complex.h -- the superbuild installs it to <prefix>/include/vcg")

if(VCG_FOUND)
    set(VCG_INCLUDE_DIRS "${VCG_INCLUDE_DIR}" "${VCG_WRAP_INCLUDE_DIR}")
    list(REMOVE_DUPLICATES VCG_INCLUDE_DIRS)
    if(NOT TARGET VCG::VCG)
        add_library(VCG::VCG INTERFACE IMPORTED)
        set_target_properties(VCG::VCG PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${VCG_INCLUDE_DIRS}")
    endif()
endif()
