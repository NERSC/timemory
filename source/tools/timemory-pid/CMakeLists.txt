
if(WIN32)
    return()
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-pid tool which provides a PID to MPI applications
#
if(TIMEMORY_BUILD_TOOLS OR TIMEMORY_BUILD_PID)

    add_executable(timemory-pid timemory-pid.cpp)

    target_link_libraries(timemory-pid PRIVATE
        timemory-compile-options
        timemory-headers
        timemory-mpi)

    set_target_properties(timemory-pid PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH ON)

    # timemory-pid installation
    install(TARGETS timemory-pid DESTINATION bin)

endif()
