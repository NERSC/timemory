
#----------------------------------------------------------------------------------------#
# Build and install timem tool
#
if(TIMEMORY_BUILD_TIMEM)
    set(_STATIC_LIB)
    foreach(_TARG hash settings manager timing-component rusage-component)
        if(TARGET timemory-${_TARG}-static AND NOT TIMEMORY_USE_COVERAGE)
            list(APPEND _STATIC_LIB timemory-${_TARG}-static)
        endif()
    endforeach()

    add_library(timem-libexplain INTERFACE)

    add_option(TIMEMORY_BUILD_TOOLS_LIBEXPECT "Enable using libexpect to diagnose errors" OFF)
    if(TIMEMORY_BUILD_TOOLS_LIBEXPECT)

        find_path(LIBEXPLAIN_INCLUDE_DIR
            NAMES libexplain/execvp.h
            PATH_SUFFIXES include)

        find_library(LIBEXPLAIN_LIBRARY
            NAMES explain
            PATH_SUFFIXES lib lib64 lib32)

        if(LIBEXPLAIN_INCLUDE_DIR AND LIBEXPLAIN_LIBRARY)
            target_include_directories(timem-libexplain INTERFACE ${LIBEXPLAIN_INCLUDE_DIR})
            target_compile_definitions(timem-libexplain INTERFACE TIMEMORY_USE_LIBEXPLAIN)
            target_link_libraries(timem-libexplain INTERFACE ${LIBEXPLAIN_LIBRARY})
        else()
            set(TIMEMORY_BUILD_TOOLS_LIBEXPECT OFF)
        endif()

    endif()

    add_executable(timem timem.cpp timem.hpp)

    target_link_libraries(timem PRIVATE
        timemory-compile-options
        timemory-arch
        timemory-vector
        timemory-headers
        timemory-papi
        timem-libexplain
        ${_STATIC_LIB})

    set_target_properties(timem PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)

    # timem installation
    install(TARGETS timem DESTINATION bin)

endif()
