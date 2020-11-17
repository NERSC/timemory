
if(TIMEMORY_BUILD_PYTHON AND TIMEMORY_BUILD_PYTHON_LINE_PROFILER)
    checkout_git_submodule(RECURSIVE
        RELATIVE_PATH external/line-profiler
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        REPO_URL https://github.com/jrmadsen/line_profiler.git
        REPO_BRANCH timemory)

    message(STATUS "Adding external/line-profiler")
    add_subdirectory(line-profiler)

    if(TARGET libpytimemory AND TARGET _line_profiler)
        add_dependencies(libpytimemory _line_profiler)
    endif()
endif()
