
##########################################################################################
#
#        timemory tools (stand-alone executables and libraries)
#
##########################################################################################

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tools)
set(CMAKE_UNITY_BUILD OFF)

#----------------------------------------------------------------------------------------#
# Build and install timemory-jump library
#
message(STATUS "Adding source/tools/timemory-jump...")
add_subdirectory(timemory-jump)

#----------------------------------------------------------------------------------------#
# Build and install timemory-stubs library
#
message(STATUS "Adding source/tools/timemory-stubs...")
add_subdirectory(timemory-stubs)

#----------------------------------------------------------------------------------------#
# Build and install timemory-avail tool
#
message(STATUS "Adding source/tools/timemory-avail...")
add_subdirectory(timemory-avail)

#----------------------------------------------------------------------------------------#
# Build and install timemory-pid tool
#
message(STATUS "Adding source/tools/timemory-pid...")
add_subdirectory(timemory-pid)

#----------------------------------------------------------------------------------------#
# Build and install timem tool
#
message(STATUS "Adding source/tools/timem...")
add_subdirectory(timem)

#----------------------------------------------------------------------------------------#
# Build and install timemory-run
#
message(STATUS "Adding source/tools/timemory-run...")
add_subdirectory(timemory-run)

#----------------------------------------------------------------------------------------#
# Build and install timemory-mpip library
#
message(STATUS "Adding source/tools/timemory-mpip...")
add_subdirectory(timemory-mpip)

#----------------------------------------------------------------------------------------#
# Build and install timemory-ompt library
#
message(STATUS "Adding source/tools/timemory-ompt...")
add_subdirectory(timemory-ompt)

#----------------------------------------------------------------------------------------#
# Build and install timemory-ncclp library
#
message(STATUS "Adding source/tools/timemory-ncclp...")
add_subdirectory(timemory-ncclp)

#----------------------------------------------------------------------------------------#
# Build and install timemory-connector libraries for kokkos
#
message(STATUS "Adding source/tools/kokkos-connector...")
add_subdirectory(kokkos-connector)
