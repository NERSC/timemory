
if(WIN32)
    set_directory_properties(PROPERTIES
        COMPILE_DEFINITIONS _TIMEMORY_DLL)
endif(WIN32)

set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

return()

file(GLOB cxx_headers ${CMAKE_CURRENT_LIST_DIR}/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/impl/*.icpp)

set(cxx_sources)
foreach(_SOURCE ${cxx_headers})
    string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/" "" _SOURCE "${_SOURCE}")
    list(APPEND cxx_sources ${_SOURCE})
endforeach()
message(STATUS "${cxx_sources}")

add_library(${LIBNAME}-cxx-library INTERFACE)

target_include_directories(${LIBNAME}-cxx-library
    INTERFACE include)

target_link_libraries(${LIBNAME}-cxx-library
    INTERFACE ${EXTERNAL_LIBRARIES})

target_compile_definitions(${LIBNAME}-cxx-library
    INTERFACE ${LIBRARY_COMPILE_DEFINITIONS})

target_sources(${LIBNAME}-cxx-library INTERFACE ${cxx_sources})

#target_compile_options(${LIBRARY_TARGET_NAME}
#    PUBLIC
#        $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
#        $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>
#        $<$<COMPILE_LANGUAGE:CUDA>:${${PROJECT_NAME}_CUDA_FLAGS}>)
