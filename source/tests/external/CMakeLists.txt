
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

if(NOT TIMEMORY_SKIP_BUILD AND TIMEMORY_BUILD_C)
    if(BUILD_SHARED_LIBS)
        add_library(custom-record-functions ${_EXCLUDE} SHARED
    	    custom_record_functions.cpp)
        target_link_libraries(custom-record-functions
            timemory::timemory-cxx-shared
            timemory::timemory-c-shared
            timemory::timemory-plotting
            common-test-libs)
    elseif(BUILD_STATIC_LIBS)
        add_library(custom-record-functions ${_EXCLUDE} STATIC
            custom_record_functions.cpp)
        target_link_libraries(custom-record-functions PUBLIC
            timemory::timemory-cxx-static
            timemory::timemory-c-static
            timemory::timemory-plotting
            common-test-libs)
    endif()
endif()

if(TIMEMORY_USE_NCCL AND TIMEMORY_BUILD_NCCLP_LIBRARY)
    add_subdirectory(nccl-tests)
endif()
