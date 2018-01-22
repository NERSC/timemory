
add_subdirectory(ex1)
if(USE_MPI)
    add_subdirectory(ex2)
endif(USE_MPI)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_BINARY_DIR}/examples)

set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})
