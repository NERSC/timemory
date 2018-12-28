
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

set_target_properties(${TIMEM_NAME} PROPERTIES
    COMPILE_FLAGS ${${PROJECT_NAME}_CXX_FLAGS})

# timem installation
install(TARGETS ${TIMEM_NAME}
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT development)
