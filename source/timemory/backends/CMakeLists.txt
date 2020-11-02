
if(TIMEMORY_USE_MPI_INIT)

    file(GLOB_RECURSE mpi_header_files ${CMAKE_CURRENT_SOURCE_DIR}/types/mpi/*.hpp)
    file(GLOB_RECURSE mpi_source_files ${CMAKE_CURRENT_SOURCE_DIR}/types/mpi/*.cpp)

    build_intermediate_library(
        NAME                mpi_init
        TARGET              mpi_init
        CATEGORY            EXTERN
        FOLDER              components
        HEADERS             ${mpi_header_files}
        SOURCES             ${mpi_source_files}
        DEPENDS             timemory-mpi timemory-core
        PROPERTY_DEPENDS    GLOBAL)

endif()
