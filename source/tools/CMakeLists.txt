
################################################################################
#
#        timem tool
#
################################################################################

# disabled
if(NOT TIMEMORY_BUILD_TOOLS)
    return()
endif()

# Windows not supported
if(WIN32)
    return()
endif()

set(TIMEM_NAME timem)

add_executable(${TIMEM_NAME} timem.cpp)

target_link_libraries(${TIMEM_NAME} timemory-cxx-compile-options timemory-headers
    timemory-extensions timemory-cxx-static)
target_compile_definitions(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_DEFINITIONS})
target_compile_options(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_CXX_FLAGS})
target_compile_definitions(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_DEFINITIONS})
target_include_directories(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})
target_include_directories(${TIMEM_NAME} SYSTEM PRIVATE ${${PROJECT_NAME}_SYSTEM_INCLUDE_DIRS})

# timem installation
install(TARGETS ${TIMEM_NAME}
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development)
