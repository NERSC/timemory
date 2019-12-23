
##########################################################################################
#
#        timem tool
#
##########################################################################################

add_executable(timemory-avail
    ${CMAKE_CURRENT_LIST_DIR}/available.cpp
    ${CMAKE_CURRENT_LIST_DIR}/available.hpp)
target_include_directories(timemory-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(timemory-avail PRIVATE timemory-extensions timemory-headers)
set_target_properties(timemory-avail PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS timemory-avail DESTINATION bin)

# disabled
if(NOT TIMEMORY_BUILD_TOOLS)
    return()
endif()

# Windows not supported
if(WIN32)
    return()
endif()

add_option(TIMEMORY_BUILD_TIMEM "Build the timem tool" ON)
if(TIMEMORY_USE_MPI)
    add_option(TIMEMORY_BUILD_MPIP "Build the mpiP library" ON)
endif()

# pmpi tool
if(TARGET timemory-cxx-shared AND TIMEMORY_USE_MPI)
    add_subdirectory(mpip)
endif()

if(NOT TIMEMORY_BUILD_TIMEM)
    return()
endif()

#----------------------------------------------------------------------------------------#
# Embed the link path into the executable
#
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

#----------------------------------------------------------------------------------------#
# Build, link, and install exe
#
add_executable(timem timem.cpp)

target_link_libraries(timem PRIVATE
    timemory-compile-options
    timemory-arch
    timemory-vector
    timemory-headers
    timemory-papi
    timem-libexplain)

set_target_properties(timem PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON)

# timem installation
install(TARGETS timem DESTINATION bin)
