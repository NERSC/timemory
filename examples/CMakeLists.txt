
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

    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/examples
        ERROR_QUIET)

    set(example_dirs ex1 ex2)
    foreach(example_dir ${example_dirs})
        execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_LIST_DIR}/${example_dir}
            ${CMAKE_BINARY_DIR}/examples/${example_dir}
            ERROR_QUIET)
    endforeach(example_dir ${example_dirs})

endif(BUILD_EXAMPLES)

set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})
