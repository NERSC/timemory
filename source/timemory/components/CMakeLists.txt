
#add_subdirectory(base)

add_subdirectory(allinea)
add_subdirectory(caliper)
add_subdirectory(cuda)
add_subdirectory(cupti)
add_subdirectory(craypat)
add_subdirectory(data_tracker)
add_subdirectory(gotcha)
add_subdirectory(gperftools)
add_subdirectory(likwid)
add_subdirectory(ompt)
add_subdirectory(rusage)
add_subdirectory(papi)
add_subdirectory(roofline)
add_subdirectory(tau_marker)
add_subdirectory(timing)
add_subdirectory(trip_count)
add_subdirectory(vtune)

if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/factory.cpp PROPERTIES
        # LANGUAGE CUDA
        LINKER_LANGUAGE CUDA)
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
    DEPENDS             timemory-external
    PROPERTY_DEPENDS    GLOBAL COMPONENT)

# should always be last component
add_subdirectory(user_bundle)
