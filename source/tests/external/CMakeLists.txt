
if(NOT TIMEMORY_BUILD_GOOGLE_TEST)
	return()
endif()

set(_EXCLUDE)
if(NOT TIMEMORY_BUILD_TESTING)
    set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

add_executable(socket-server ${_EXCLUDE} server.cpp)
add_executable(socket-client ${_EXCLUDE} client.cpp)
add_library(extern-test-templates ${_EXCLUDE} SHARED extern_templates.cpp)
add_library(gotcha-tests-lib ${_EXCLUDE} SHARED gotcha_tests_lib.hpp gotcha_tests_lib.cpp)

target_include_directories(gotcha-tests-lib PUBLIC ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(socket-server PRIVATE common-test-libs)
target_link_libraries(socket-client PRIVATE common-test-libs)
target_link_libraries(extern-test-templates PUBLIC common-test-libs)
target_link_libraries(gotcha-tests-lib PUBLIC common-test-libs)

set(_BUILD OFF)
if(BUILD_SHARED_LIBS)
    set(_BUILD ON)
    set(_LIB_TAG shared)
    set(_LIB_TYPE SHARED)
elseif(BUILD_STATIC_LIBS)
    set(_BUILD ON)
    set(_LIB_TAG static)
    set(_LIB_TYPE STATIC)
endif()

if(NOT TIMEMORY_SKIP_BUILD AND _BUILD)
    if(TIMEMORY_BUILD_C)
        add_library(custom-record-functions ${_EXCLUDE} ${_LIB_TYPE}
            custom_record_functions.cpp)
        target_link_libraries(custom-record-functions PUBLIC
            timemory::timemory-cxx-${_LIB_TAG}
            timemory::timemory-c-${_LIB_TAG}
            timemory::timemory-plotting
            common-test-libs)
    endif()

    if(TIMEMORY_USE_DYNINST)
        add_library(custom-dynamic-instr ${_EXCLUDE} ${_LIB_TYPE}
            custom_dynamic_instr.cpp)
        target_link_libraries(custom-dynamic-instr PUBLIC
            timemory::timemory-cxx-${_LIB_TAG}
            timemory::timemory-plotting
            common-test-libs)
        set_target_properties(custom-dynamic-instr PROPERTIES
            OUTPUT_NAME                 custom_dynamic_instr
            LIBRARY_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY    ${PROJECT_BINARY_DIR})
    endif()
endif()

if(TIMEMORY_USE_NCCL AND TIMEMORY_BUILD_NCCLP_LIBRARY)
    add_subdirectory(nccl-tests)
endif()
