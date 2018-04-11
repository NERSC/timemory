
project(TiMemory-Examples C CXX)

if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    option(TIMEMORY_USE_MPI "Enable MPI usage" ON)
    if(TIMEMORY_USE_MPI)
        find_package(MPI)
    endif(TIMEMORY_USE_MPI)

    add_subdirectory(ex1)
    if(TIMEMORY_USE_MPI AND MPI_FOUND)
        add_subdirectory(ex2)
    endif(TIMEMORY_USE_MPI AND MPI_FOUND)


else("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")

    if(TIMEMORY_BUILD_EXAMPLES)
        add_subdirectory(ex1)
        if(TIMEMORY_USE_MPI AND MPI_FOUND)
            add_subdirectory(ex2)
        endif(TIMEMORY_USE_MPI AND MPI_FOUND)

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

    endif(TIMEMORY_BUILD_EXAMPLES)

    set_property(GLOBAL APPEND PROPERTY
        BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

endif("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")


