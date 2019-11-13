
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY )
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY )
unset(CMAKE_PDB_OUTPUT_DIRECTORY )

set(DEVICE_LANGUAGE CXX)
if(TIMEMORY_USE_CUDA)
    set(DEVICE_LANGUAGE CUDA)
endif()

# add_library(no-store-merge INTERFACE)
# add_target_cxx_flag_if_avail(no-store-merge "-fno-store-merging")

set(HYBRID_COMPONENTS )

if(TIMEMORY_USE_CUDA)
    list(APPEND HYBRID_COMPONENTS timemory-cuda timemory-cudart)
endif()

if(TIMEMORY_USE_PAPI)
    list(APPEND HYBRID_COMPONENTS timemory-papi)
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_cuda_LIBRARY)
    list(APPEND HYBRID_COMPONENTS timemory-cupti timemory-cudart-device)
endif()

if(TIMEMORY_USE_CALIPER)
    list(APPEND HYBRID_COMPONENTS timemory-caliper)
endif()

message(STATUS "[${PROJECT_NAME}] hybrid test components: ${HYBRID_COMPONENTS}")

add_subdirectory(external)

add_timemory_google_test(mangle_tests
    DISCOVER_TESTS
    SOURCES         mangle_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)

add_timemory_google_test(tuple_tests
    DISCOVER_TESTS
    SOURCES         tuple_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-papi timemory-analysis-tools timemory-caliper)

add_timemory_google_test(macro_tests
    DISCOVER_TESTS
    SOURCES         macro_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)

add_timemory_google_test(hybrid_tests
    DISCOVER_TESTS
    SOURCES         hybrid_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    ${HYBRID_COMPONENTS} timemory-analysis-tools)

add_timemory_google_test(timing_tests
    DISCOVER_TESTS
    SOURCES         timing_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)

add_timemory_google_test(rusage_tests
    DISCOVER_TESTS
    SOURCES         rusage_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)

if(TIMEMORY_USE_PAPI)
    add_timemory_google_test(papi_tests
        DISCOVER_TESTS
        SOURCES         papi_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-papi timemory-compile-options
                        timemory-develop-options timemory-analysis-tools)
endif()

add_timemory_google_test(apply_tests
    DISCOVER_TESTS
    SOURCES         apply_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)

if(TIMEMORY_USE_ARCH)
    add_timemory_google_test(aligned_allocator_tests
        DISCOVER_TESTS
        SOURCES         aligned_allocator_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch timemory-compile-options
                        timemory-develop-options timemory-analysis-tools)
endif()

set_source_files_properties(cuda_tests.cpp cupti_tests.cpp
    PROPERTIES
    LANGUAGE        ${DEVICE_LANGUAGE}
    LINKER_LANGUAGE ${DEVICE_LANGUAGE})

if(TIMEMORY_USE_CUDA)
    add_timemory_google_test(cuda_tests
        DISCOVER_TESTS
        SOURCES         cuda_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options
                        timemory-cuda    timemory-cudart timemory-develop-options
                        timemory-analysis-tools)
endif()

if(TIMEMORY_USE_CUDA AND TIMEMORY_USE_NVTX)
    add_timemory_google_test(cuda_nvtx_tests
        DISCOVER_TESTS
        SOURCES         cuda_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options timemory-develop-options
                        timemory-cuda    timemory-cudart timemory-nvtx
                        timemory-analysis-tools)
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_cuda_LIBRARY)
    add_library(cupti-compile-options INTERFACE)
    target_compile_options(cupti-compile-options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
    add_timemory_google_test(cupti_tests
        DISCOVER_TESTS
        SOURCES         cupti_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options timemory-develop-options
                        timemory-cuda    timemory-cupti  # timemory-cudart-device
                        cupti-compile-options timemory-analysis-tools)
endif()

if(_HAS_PROFILER)
    add_timemory_google_test(gperf_cpu_tests
        SOURCES         gperf_cpu_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                        timemory-gperftools-cpu timemory-analysis-tools
        ENVIRONMENT     "CPUPROFILE_FREQUENCY=1000;CPUPROFILE_REALTIME=1")
endif()

add_timemory_google_test(gperf_fake_cpu_tests
    DISCOVER_TESTS
    SOURCES         gperf_cpu_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-analysis-tools)

if(_HAS_TCMALLOC)
    add_timemory_google_test(gperf_heap_tests
        DISCOVER_TESTS
        SOURCES         gperf_heap_tests.cpp
        LINK_LIBRARIES  timemory-headers    timemory-arch   timemory-compile-options timemory-develop-options
                        timemory-gperftools-heap timemory-analysis-tools)
endif()

if(TIMEMORY_USE_GOTCHA)
    add_timemory_google_test(gotcha_tests
        DISCOVER_TESTS
        SOURCES         gotcha_tests.cpp
        LINK_LIBRARIES  timemory-headers    timemory-compile-options timemory-develop-options
                        timemory-gotcha     timemory-mpi
                        timemory-papi       timemory-vector
                        timemory-arch       gotcha-tests-lib
                        timemory-analysis-tools)
endif()

add_timemory_google_test(priority_tests
    DISCOVER_TESTS
    SOURCES         priority_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)
