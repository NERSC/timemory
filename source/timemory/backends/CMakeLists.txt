
file(GLOB_RECURSE mpi_header_files ${CMAKE_CURRENT_SOURCE_DIR}/types/mpi/*.hpp)
file(GLOB_RECURSE mpi_source_files ${CMAKE_CURRENT_SOURCE_DIR}/types/mpi/*.cpp)
file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(TIMEMORY_USE_MPI_INIT)

    build_intermediate_library(
        NAME                mpi_init
        TARGET              common
        CATEGORY            GLOBAL
        FOLDER              components
        HEADERS             ${mpi_header_files}
        SOURCES             ${mpi_source_files}
        DEPENDS             timemory-mpi timemory-core
        PROPERTY_DEPENDS    GLOBAL)

endif()

foreach(_category header source)
    list(REMOVE_ITEM ${_category}_files ${mpi_${_category}_files})
endforeach()

build_intermediate_library(
    USE_INTERFACE
    INSTALL_SOURCES
    NAME                backends
    TARGET              common
    CATEGORY            GLOBAL
    FOLDER              components
    HEADERS             ${header_files}
    SOURCES             ${source_files}
    DEPENDS             timemory-mpi timemory-core
    PROPERTY_DEPENDS    GLOBAL)
