#------------------------------------------------------------------------------#
#
#       Finds headers and libraries for ScoreP instrumentation
#
#------------------------------------------------------------------------------#

include(FindPackageHandleStandardArgs)

#------------------------------------------------------------------------------#

set(ScoreP_HINTS "$ENV{ScoreP_ROOT}")

#------------------------------------------------------------------------------#

find_path(ScoreP_ROOT_DIR
    NAMES include/scorep/SCOREP_User.h
    HINTS ${ScoreP_HINTS}
    PATHS ${ScoreP_HINTS}
)

#------------------------------------------------------------------------------#

find_path(ScoreP_INCLUDE_DIR
    NAMES scorep/SCOREP_User.h
    PATH_SUFFIXES include
    HINTS ${ScoreP_ROOT_DIR} ${ScoreP_HINTS}
    PATHS ${ScoreP_ROOT_DIR} ${ScoreP_HINTS}
)

#------------------------------------------------------------------------------#

if(ScoreP_INCLUDE_DIR)
    set(ScoreP_INCLUDE_DIRS ${ScoreP_INCLUDE_DIR})
endif()

#------------------------------------------------------------------------------#

set(ScoreP_AVAILABLE_COMPONENTS
    #
    #   COMPILER
    #
    #scorep_adapter_compiler_event
    #scorep_adapter_compiler_mgmt
    #
    #   MEMORY
    #
    #scorep_adapter_memory_event_cxx
    #scorep_adapter_memory_event_cxx14_L32
    #scorep_adapter_memory_event_cxx14_L64
    #scorep_adapter_memory_event_cxx_L32
    #scorep_adapter_memory_event_cxx_L64
    #scorep_adapter_memory_event_hbwmalloc
    #scorep_adapter_memory_event_libc
    #scorep_adapter_memory_event_libc11
    #scorep_adapter_memory_event_pgCC
    #scorep_adapter_memory_event_pgCC_L32
    #scorep_adapter_memory_event_pgCC_L64
    #scorep_adapter_memory_mgmt
    #
    #   MPI
    #
    scorep_adapter_mpi_event
    scorep_adapter_mpi_mgmt
    #
    #   OPARI2
    #
    #scorep_adapter_opari2_mgmt
    #scorep_adapter_opari2_openmp_event
    #scorep_adapter_opari2_openmp_mgmt
    #scorep_adapter_opari2_user_event
    #scorep_adapter_opari2_user_mgmt
    #
    #   OPENCL
    #
    #scorep_adapter_opencl_event_linktime
    #scorep_adapter_opencl_event_runtime
    #scorep_adapter_opencl_mgmt_linktime
    #scorep_adapter_opencl_mgmt_runtime
    #
    #   POSIX_IO
    #
    #scorep_adapter_posix_io_event_linktime
    #scorep_adapter_posix_io_event_runtime
    #scorep_adapter_posix_io_mgmt_linktime
    #scorep_adapter_posix_io_mgmt_runtime
    #
    #   PTHREAD
    #
    scorep_adapter_pthread_event
    scorep_adapter_pthread_mgmt
    #
    #   USER
    #
    scorep_adapter_user_event
    scorep_adapter_user_mgmt
    #
    #   MISC
    #
    #scorep_adapter_utils
    #scorep_alloc_metric
    scorep_constructor
    #scorep_estimator
    scorep_measurement
    #
    #   MPP
    #
    #scorep_mpp_mockup
    scorep_mpp_mpi
    #
    #   MUTEX
    #
    #scorep_mutex_mockup
    #scorep_mutex_omp
    scorep_mutex_pthread
    #scorep_mutex_pthread_spinlock
    #scorep_mutex_pthread_wrap
    #
    #   ONLINE
    #
    #scorep_online_access_mockup
    #scorep_online_access_mpp_mpi
    scorep_online_access_spp
    #
    #   THREAD
    #
    scorep_thread_create_wait_pthread
    #scorep_thread_fork_join_omp
    #scorep_thread_mockup
    #
    #   DEPENDS
    #
    m
    dl
    z
    unwind
)

#------------------------------------------------------------------------------#

if(NOT ScoreP_FIND_COMPONENTS)
    set(ScoreP_FIND_COMPONENTS ${ScoreP_AVAILABLE_COMPONENTS})
elseif("all" IN_LIST ScoreP_FIND_COMPONENTS)
    list(APPEND ScoreP_FIND_COMPONENTS ${ScoreP_AVAILABLE_COMPONENTS})
    list(REMOVE_DUPLICATES ScoreP_FIND_COMPONENTS)
endif()

#------------------------------------------------------------------------------#

foreach(_COMP ${ScoreP_FIND_COMPONENTS})

    if(NOT "${_COMP}" IN_LIST ScoreP_AVAILABLE_COMPONENTS)
        message(AUTHOR_WARNING "Non-standard ScoreP component: '${_COMP}'")
    endif()

    find_library(ScoreP_${_COMP}_LIBRARY
        NAMES ${_COMP}
        PATH_SUFFIXES lib64 lib lib32
        HINTS ${ScoreP_ROOT_DIR} ${ScoreP_LIBRARY_DIR} ${ScoreP_HINTS}
        PATHS ${ScoreP_ROOT_DIR} ${ScoreP_LIBRARY_DIR} ${ScoreP_HINTS}
    )

    if(ScoreP_${_COMP}_LIBRARY)
        list(APPEND ScoreP_LIBRARIES ${ScoreP_${_COMP}_LIBRARY})
        get_filename_component(_COMP_DIR "${ScoreP_${_COMP}_LIBRARY}" PATH)
	if(NOT ScoreP_LIBRARY_DIR)
            set(ScoreP_LIBRARY_DIR ${_COMP_DIR})
	endif()
        list(APPEND ScoreP_LIBRARY_DIRS ${_COMP_DIR})
    else()
        if(ScoreP_FIND_REQUIRED_${_COMP})
            list(APPEND _ScoreP_MISSING_COMPONENTS ScoreP_${_COMP}_LIBRARY)
        endif()
    endif()
    mark_as_advanced(ScoreP_${_COMP}_LIBRARY)

endforeach()

#------------------------------------------------------------------------------#

mark_as_advanced(ScoreP_INCLUDE_DIR ScoreP_LIBRARY ScoreP_LIBRARY_DIR)

find_package_handle_standard_args(ScoreP REQUIRED_VARS
    ScoreP_ROOT_DIR ScoreP_INCLUDE_DIR ScoreP_LIBRARY_DIR
    ${_ScoreP_MISSING_COMPONENTS})

#------------------------------------------------------------------------------#

add_library(ScoreP::scorep IMPORTED UNKNOWN)

if(ScoreP_INCLUDE_DIRS AND ScoreP_LIBRARIES)
    target_include_directories(ScoreP::scorep INTERFACE ${ScoreP_INCLUDE_DIR})
    target_link_libraries(ScoreP::scorep INTERFACE ${ScoreP_LIBRARIES})
    target_compile_definitions(ScoreP::scorep INTERFACE SCOREP_USER_ENABLE)
endif()

#------------------------------------------------------------------------------#
