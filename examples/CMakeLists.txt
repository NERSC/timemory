
if(WIN32)
    add_option(BUILD_EXAMPLES "Build the C++ examples" OFF)
else(WIN32)
    add_option(BUILD_EXAMPLES "Build the C++ examples" ON)
endif(WIN32)

if(BUILD_EXAMPLES)
    add_subdirectory(ex1)
    if(USE_MPI AND MPI_FOUND)
        add_subdirectory(ex2)
    endif(USE_MPI AND MPI_FOUND)
endif(BUILD_EXAMPLES)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_BINARY_DIR}/examples)

set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})
