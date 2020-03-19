
if(TIMEMORY_USE_PAPI OR TIMEMORY_USE_CUPTI)
    set(NAME roofline)
    set(DEPS
        timemory-papi-component timemory-cupti-component
        timemory-cpu-roofline timemory-gpu-roofline timemory-roofline)

    file(GLOB_RECURSE header_files ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
    file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                                   ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

    build_intermediate_library(
        NAME                ${NAME}
        TARGET              ${NAME}-component
        CATEGORY            COMPONENT
        FOLDER              components
        HEADERS             ${header_files}
        SOURCES             ${source_files}
        DEPENDS             ${DEPS}
        PROPERTY_DEPENDS    GLOBAL)
endif()
