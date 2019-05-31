
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

target_link_libraries(${TIMEM_NAME} ${LIBNAME}-cxx-library-static timemory-headers)
target_compile_definitions(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_DEFINITIONS})
target_compile_options(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_CXX_FLAGS})
target_include_directories(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

# timem installation
install(TARGETS ${TIMEM_NAME}
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development)
