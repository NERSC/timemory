
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

#----------------------------------------------------------------------------------------#
# PAPI interface library with static linkage if possible
#
add_library(timem-papi INTERFACE)
if(PAPI_FOUND)
    target_compile_definitions(timem-papi INTERFACE TIMEMORY_USE_PAPI)
    target_include_directories(timem-papi INTERFACE ${PAPI_INCLUDE_DIRS})
    if(PAPI_STATIC_LIBRARY AND PAPI_pfm_LIBRARY)
        target_link_libraries(timem-papi INTERFACE ${PAPI_STATIC_LIBRARY} ${PAPI_pfm_LIBRARY})
    else()
        target_link_libraries(timem-papi INTERFACE ${PAPI_LIBRARIES})
    endif()
endif()

#----------------------------------------------------------------------------------------#
# Build, link, and install exe
#
add_executable(timem timem.cpp)

target_link_libraries(timem PRIVATE
    timemory-compile-options
    timemory-headers
    timemory-cxx-static
    timem-papi)

# timem installation
install(TARGETS timem
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development)
