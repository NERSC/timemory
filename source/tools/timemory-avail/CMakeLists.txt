
if(NOT TIMEMORY_BUILD_AVAIL)
  set(_EXCLUDE EXCLUDE_FROM_ALL)
  set(_OPTIONAL OPTIONAL)
endif()

add_executable(timemory-avail ${_EXCLUDE}
    ${CMAKE_CURRENT_LIST_DIR}/timemory-avail.cpp
    ${CMAKE_CURRENT_LIST_DIR}/timemory-avail.hpp)
target_include_directories(timemory-avail PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(timemory-avail PRIVATE timemory-extensions timemory-headers)
set_target_properties(timemory-avail PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
install(TARGETS timemory-avail
    DESTINATION bin
    COMPONENT tools
    ${_OPTIONAL})
