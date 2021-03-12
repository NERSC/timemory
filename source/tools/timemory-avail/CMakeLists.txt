
if(NOT TIMEMORY_BUILD_AVAIL)
  set(_EXCLUDE EXCLUDE_FROM_ALL)
endif()

add_executable(timemory-avail ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/timemory-avail.cpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory-avail.hpp)

target_include_directories(timemory-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(timemory-avail PRIVATE
    timemory::timemory-compile-options
    timemory::timemory-compile-timing
    timemory::timemory-extensions
    timemory::timemory-headers
    timemory::timemory-python)

set_target_properties(timemory-avail PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)

install(TARGETS timemory-avail
    DESTINATION bin
    COMPONENT tools
    OPTIONAL)
