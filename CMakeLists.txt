cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include(ExternalProject)

project(Photogrammetry)

ExternalProject_Add(
	VCG
	GIT_REPOSITORY https://github.com/cdcseacave/VCG
	GIT_TAG 56c76b08893a556e8bbd70673d84a82777b8cfff
	BUILD_COMMAND ""
	CONFIGURE_COMMAND ""
	INSTALL_COMMAND ""
)
ExternalProject_Add(
	openMVG
	GIT_REPOSITORY https://github.com/openMVG/openMVG
	GIT_TAG 835f1e585377a55d4143548fa9181438f5d23889
	SOURCE_SUBDIR src
	INSTALL_COMMAND printf "iPhone X\\\\\\\\;4.8\\nILCE-6300\\\\\\\\;23.5\\n" >> ../openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt
)
ExternalProject_Add(
	openMVS
	GIT_REPOSITORY https://github.com/cdcseacave/openMVS
	GIT_TAG 063e5cd7f1139aceaf882f9f6ae3f0de26c59138
	CMAKE_ARGS -DVCG_ROOT=${CMAKE_BINARY_DIR}/VCG-prefix/src/VCG -DCMAKE_BUILD_TYPE=Release
	DEPENDS VCG
	INSTALL_COMMAND ""
)
