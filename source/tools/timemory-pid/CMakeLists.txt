
if(WIN32)
    return()
endif()

#----------------------------------------------------------------------------------------#
# Build and install timemory-pid tool which provides a PID to MPI applications
#
add_executable(timemory-pid EXCLUDE_FROM_ALL timemory-pid.cpp)

target_link_libraries(timemory-pid PRIVATE
    timemory::timemory-compile-options
    timemory::timemory-headers
    timemory::timemory-mpi)

set_target_properties(timemory-pid PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON)

# timemory-pid installation
install(TARGETS timemory-pid DESTINATION bin OPTIONAL)
