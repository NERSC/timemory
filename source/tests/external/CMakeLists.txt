
if(NOT TIMEMORY_BUILD_GTEST)
	return()
endif()

add_library(gotcha-tests-lib SHARED
    gotcha_tests_lib.hpp
    gotcha_tests_lib.cpp)
  
target_link_libraries(gotcha-tests-lib PUBLIC
    timemory-compile-options
    timemory-arch
    timemory-vector)
  
target_include_directories(gotcha-tests-lib PUBLIC
    ${CMAKE_CURRENT_LIST_DIR})
