cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include(ExternalProject)

project(Photogrammetry)

ExternalProject_Add(
	openMVG
	GIT_REPOSITORY https://github.com/openMVG/openMVG
	GIT_TAG 835f1e585377a55d4143548fa9181438f5d23889
	SOURCE_SUBDIR src
	STEP_TARGETS build
)
ExternalProject_Add(
	VCG
	GIT_REPOSITORY https://github.com/cdcseacave/VCG
	GIT_TAG 56c76b08893a556e8bbd70673d84a82777b8cfff
)
ExternalProject_Add(
	openMVS
	GIT_REPOSITORY https://github.com/cdcseacave/openMVS
	GIT_TAG 063e5cd7f1139aceaf882f9f6ae3f0de26c59138
)

set(ENV{VCG_ROOT} "$CMAKE_BIN_DIR/VCG-prefix/")
