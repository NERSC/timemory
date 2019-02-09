
################################################################################
#
#        timem tool
#
################################################################################

if(WIN32)
    # no support for Windows yet
    return()
endif(WIN32)

set(TIMEM_NAME timem)

add_executable(${TIMEM_NAME} timem.cpp)

target_link_libraries(${TIMEM_NAME} ${LIBNAME}-cxx-library ${EXTERNAL_LIBRARIES})
target_compile_definitions(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_DEFINITIONS})
target_compile_options(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_CXX_FLAGS})
target_include_directories(${TIMEM_NAME} PRIVATE ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

# timem installation
install(TARGETS ${TIMEM_NAME}
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development)
