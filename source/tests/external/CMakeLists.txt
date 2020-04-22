
if(NOT TIMEMORY_BUILD_GTEST AND NOT TIMEMORY_BUILD_TESTING)
	return()
endif()

add_library(extern-test-templates SHARED
    extern_templates.cpp)

target_link_libraries(extern-test-templates PUBLIC
    timemory-headers
    timemory-compile-options
    timemory-arch
    timemory-vector)

add_library(gotcha-tests-lib SHARED
    gotcha_tests_lib.hpp
    gotcha_tests_lib.cpp)
  
target_link_libraries(gotcha-tests-lib PUBLIC
    timemory-compile-options
    timemory-arch
    timemory-vector)
  
target_include_directories(gotcha-tests-lib PUBLIC
    ${CMAKE_CURRENT_LIST_DIR})

if(NOT TIMEMORY_SKIP_BUILD AND TIMEMORY_BUILD_C)
    set(_LINK timemory-headers timemory-compile-options timemory-develop-options
        timemory-plotting timemory-analysis-tools)

    if(BUILD_SHARED_LIBS)
        add_library(custom-record-functions SHARED
    	    custom_record_functions.cpp)
        target_link_libraries(custom-record-functions
            timemory-cxx-shared timemory-c-shared ${_LINK})
    elseif(BUILD_STATIC_LIBS)
        add_library(custom-record-functions STATIC
            custom_record_functions.cpp)
        target_link_libraries(custom-record-functions PUBLIC
            timemory-cxx-static timemory-c-static ${_LINK})
    endif()
endif()
