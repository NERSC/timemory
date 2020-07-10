
if(WIN32)
    return()
endif()

file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

build_intermediate_library(
    USE_INTERFACE
    INSTALL_SOURCE
    NAME                utility
    TARGET              utility
    CATEGORY            GLOBAL
    FOLDER              components
    HEADERS             ${header_files}
    SOURCES             ${source_files}
    PROPERTY_DEPENDS    GLOBAL)
