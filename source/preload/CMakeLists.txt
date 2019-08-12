
##########################################################################################
#
#        preload library
#
##########################################################################################

# Windows not supported
if(WIN32)
    return()
endif()

#----------------------------------------------------------------------------------------#
# Embed the link path into the executable
#
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(LINK_LIBS
    timemory-headers
    timemory-arch
    timemory-vector
    timemory-compile-options
    timemory-papi
    timemory-cuda
    timemory-cupti
    timemory-cudart-device)

# set(VARIANTS papi papi-cuda papi-cuda-cupti cuda cuda-cupti)
# set(papi_VARIANT_LINK_LIBS timemory-papi)
# set(papi-cuda_VARIANT_LINK_LIBS timemory-papi timemory-cuda timemory-cudart)
# set(papi-cuda-cupti_VARIANT_LINK_LIBS timemory-papi timemory-cuda timemory-cupti timemory-cudart-device)
# set(cuda_VARIANT_LINK_LIBS timemory-cuda timemory-cudart)
# set(cuda-cupti_VARIANT_LINK_LIBS timemory-cuda timemory-cupti timemory-cudart-device)

#----------------------------------------------------------------------------------------#
# Build, link, and install libraries
#
build_library(
    PIC
    TYPE                SHARED
    TARGET_NAME         timemory-preload-shared
    OUTPUT_NAME         timemory-preload
    SOURCES             preload_api.cpp preload_api.hpp
    LINK_LIBRARIES      PRIVATE ${LINK_LIBS} timemory-cudart)

build_library(
    TYPE                STATIC
    TARGET_NAME         timemory-preload-static
    OUTPUT_NAME         timemory-preload
    SOURCES             preload_api.cpp preload_api.hpp
    LINK_LIBRARIES      PRIVATE ${LINK_LIBS} timemory-cudart-static)

install(TARGETS timemory-preload-shared timemory-preload-static
    DESTINATION ${TIMEMORY_INSTALL_LIBDIR})

add_subdirectory(tests)
