
#add_subdirectory(base)

add_subdirectory(rusage)
add_subdirectory(timing)
add_subdirectory(trip_count)

if(TIMEMORY_USE_ALLINEA_MAP)
    add_subdirectory(allinea)
endif()

# if(TIMEMORY_USE_CRAYPAT)
    add_subdirectory(craypat)
# endif()

if(TIMEMORY_USE_CALIPER)
    add_subdirectory(caliper)
endif()

if(TIMEMORY_USE_CUDA)
    add_subdirectory(cuda)
endif()

if(TIMEMORY_USE_CUPTI)
    add_subdirectory(cupti)
endif()

if(TIMEMORY_USE_GOTCHA)
    add_subdirectory(gotcha)
endif()

if(TIMEMORY_USE_GPERFTOOLS)
    add_subdirectory(gperftools)
endif()

if(TIMEMORY_USE_LIKWID)
    add_subdirectory(likwid)
endif()

if(TIMEMORY_USE_PAPI)
    add_subdirectory(papi)
endif()

if(TIMEMORY_USE_PAPI OR TIMEMORY_USE_CUPTI)
    add_subdirectory(roofline)
endif()

if(TIMEMORY_USE_TAU)
    add_subdirectory(tau_marker)
endif()

if(TIMEMORY_USE_VTUNE)
    add_subdirectory(vtune)
endif()

if(TIMEMORY_USE_OMPT)
    add_subdirectory(ompt)
endif()

# factory generation
build_intermediate_library(
    USE_INTERFACE
    NAME                component-factory
    TARGET              component-factory
    CATEGORY            FACTORY
    FOLDER              components
    HEADERS             ${CMAKE_CURRENT_SOURCE_DIR}/factory.hpp
    SOURCES             ${CMAKE_CURRENT_SOURCE_DIR}/factory.cpp
    PROPERTY_DEPENDS    GLOBAL COMPONENT)

# should always be last component
add_subdirectory(user_bundle)
