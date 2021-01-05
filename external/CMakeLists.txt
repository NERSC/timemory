
if(TIMEMORY_BUILD_PYTHON AND TIMEMORY_BUILD_PYTHON_LINE_PROFILER)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/line-profiler
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/line_profiler.git
        REPO_BRANCH timemory)

    message(STATUS "Adding external/line-profiler...")
    add_subdirectory(line-profiler)

    if(TARGET libpytimemory AND TARGET _line_profiler)
        add_dependencies(libpytimemory _line_profiler)
    endif()
endif()

if(TIMEMORY_BUILD_PYTHON AND TIMEMORY_BUILD_PYTHON_HATCHET)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/hatchet
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/hatchet.git
        REPO_BRANCH timemory)

    # line_profiler might have already found Cython
    if(CYTHON_EXECUTABLE)
        find_package(Cython QUIET)
    endif()

    if(CYTHON_EXECUTABLE OR Cython_FOUND)
        message(STATUS "Adding external/hatchet...")
        add_subdirectory(hatchet)
    else()
        message(WARNING "Internal Hatchet could not built because Cython could not be found")
        set(TIMEMORY_BUILD_PYTHON_HATCHET OFF PARENT_SCOPE)
    endif()

    if(TARGET libpytimemory AND TARGET hatchet)
        add_dependencies(libpytimemory hatchet)
    endif()
endif()
