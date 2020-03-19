
#add_subdirectory(base)

add_subdirectory(rusage)
add_subdirectory(timing)
add_subdirectory(trip_count)

if(TIMEMORY_USE_ALLINEA)
    add_subdirectory(allinea)
endif()

if(TIMEMORY_USE_CALIPER)
    add_subdirectory(caliper)
endif()

if(TIMEMORY_USE_CUDA)
    add_subdirectory(cuda)
endif()

if(TIMEMORY_USE_CUPTI)
    add_subdirectory(cupti)
endif()

if(TIMEMORY_USE_GOTCHA)
    add_subdirectory(gotcha)
endif()

if(TIMEMORY_USE_GPERFTOOLS)
    add_subdirectory(gperftools)
endif()

if(TIMEMORY_USE_LIKWID)
    add_subdirectory(likwid)
endif()

if(TIMEMORY_USE_PAPI)
    add_subdirectory(papi)
endif()

if(TIMEMORY_USE_PAPI OR TIMEMORY_USE_CUPTI)
    add_subdirectory(roofline)
endif()

if(TIMEMORY_USE_TAU)
    add_subdirectory(tau_marker)
endif()

if(TIMEMORY_USE_VTUNE)
    add_subdirectory(vtune)
endif()

# should always be last
add_subdirectory(user_bundle)
