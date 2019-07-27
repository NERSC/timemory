
##########################################################################################
#
#        timem tool
#
##########################################################################################

# disabled
if(NOT TIMEMORY_BUILD_TOOLS)
    return()
endif()

# Windows not supported
if(WIN32)
    return()
endif()

#----------------------------------------------------------------------------------------#
# Embed the link path into the executable
#
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
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
# PAPI interface library with static linkage if possible
#
add_library(timem-papi INTERFACE)
if(PAPI_FOUND)
    #target_compile_definitions(timem-papi INTERFACE TIMEMORY_USE_PAPI)
    #target_include_directories(timem-papi INTERFACE ${PAPI_INCLUDE_DIRS})
    #if(TARGET timemory-papi-static)
    #    target_link_libraries(timem-papi INTERFACE timemory-papi-static)
    #else()
        target_link_libraries(timem-papi INTERFACE timemory-papi)
    #endif()
endif()

#----------------------------------------------------------------------------------------#
# Build, link, and install exe
#
add_executable(timem timem.cpp)

target_link_libraries(timem PRIVATE
    timemory-compile-options
    timemory-headers
    timemory-cxx-static
    timem-papi
    timem-libexplain)

#set_target_properties(timem PROPERTIES POSITION_INDEPENDENT_CODE OFF)
  
# timem installation
install(TARGETS timem
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development)
