
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY )
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY )
unset(CMAKE_PDB_OUTPUT_DIRECTORY )

set(DEVICE_LANGUAGE CXX)
if(TIMEMORY_USE_CUDA)
    set(DEVICE_LANGUAGE CUDA)
endif()

option(TIMEMORY_USE_TESTING_LINK_LIBRARY "Google-tests like to shared or static library" ON)
mark_as_advanced(TIMEMORY_USE_TESTING_LINK_LIBRARY)

set(_LIBRARY)
if(TIMEMORY_USE_TESTING_LINK_LIBRARY)
    if(TARGET timemory-cxx-shared)
        set(_LIBRARY timemory-cxx-shared)
    elseif(TARGET timemory-cxx-static)
        set(_LIBRARY timemory-cxx-static)
    endif()
endif()

if(TIMEMORY_USE_OMPT)
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
        set(_OPENMP OpenMP::OpenMP_CXX)
    endif()
endif()

# add_library(no-store-merge INTERFACE)
# add_target_cxx_flag_if_avail(no-store-merge "-fno-store-merging")
add_library(test-opt-flags INTERFACE)
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
target_compile_options(test-opt-flags INTERFACE ${_FLAGS})
if(NOT TIMEMORY_USE_ARCH)
  add_target_flag_if_avail(test-opt-flags "-march=native")
endif()

set(HYBRID_COMPONENTS )

if(TIMEMORY_USE_CUDA)
    list(APPEND HYBRID_COMPONENTS timemory-cuda timemory-cudart)
endif()

if(TIMEMORY_USE_PAPI)
    list(APPEND HYBRID_COMPONENTS timemory-papi)
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_DRIVER_LIBRARY)
    list(APPEND HYBRID_COMPONENTS timemory-cupti timemory-cudart-device)
endif()

if(TIMEMORY_USE_CALIPER)
    list(APPEND HYBRID_COMPONENTS timemory-caliper)
endif()

if(TIMEMORY_USE_VTUNE)
    list(APPEND HYBRID_COMPONENTS timemory-vtune)
endif()

set(ENV{TIMEMORY_ENABLED} OFF)

add_subdirectory(external)

if(TARGET custom-record-functions)
    add_timemory_google_test(library_tests
        DISCOVER_TESTS
        SOURCES         library_tests.cpp
        LINK_LIBRARIES  custom-record-functions)
endif()

add_timemory_google_test(argparse_tests
    SOURCES         argparse_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-analysis-tools)

add_timemory_google_test(flat_tests
    DISCOVER_TESTS
    SOURCES         flat_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools extern-test-templates)

add_timemory_google_test(timeline_tests
    DISCOVER_TESTS
    SOURCES         timeline_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools extern-test-templates)

add_timemory_google_test(data_tracker_tests
    DISCOVER_TESTS
    SOURCES         data_tracker_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools)

add_timemory_google_test(threading_tests
    DISCOVER_TESTS
    SOURCES         threading_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools timemory-ompt
                    ${_LIBRARY} ${_OPENMP})

add_timemory_google_test(stl_overload_tests
    DISCOVER_TESTS
    SOURCES         stl_overload_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

add_timemory_google_test(user_bundle_tests
    DISCOVER_TESTS
    SOURCES         user_bundle_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

add_timemory_google_test(mangle_tests
    DISCOVER_TESTS
    SOURCES         mangle_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

add_timemory_google_test(tuple_tests
    DISCOVER_TESTS
    SOURCES         tuple_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-papi timemory-plotting timemory-analysis-tools timemory-caliper
                    ${_LIBRARY})

add_timemory_google_test(macro_tests
    DISCOVER_TESTS
    SOURCES         macro_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

add_timemory_google_test(hybrid_tests
    DISCOVER_TESTS
    SOURCES         hybrid_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    ${HYBRID_COMPONENTS} timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

add_timemory_google_test(timing_tests
    DISCOVER_TESTS
    SOURCES         timing_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

add_timemory_google_test(rusage_tests
    DISCOVER_TESTS
    SOURCES         rusage_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

if(TIMEMORY_USE_PAPI)
    add_timemory_google_test(papi_tests
        DISCOVER_TESTS
        SOURCES         papi_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-papi timemory-compile-options
                        timemory-develop-options timemory-plotting timemory-analysis-tools
                        timemory-arch timemory-roofline-options test-opt-flags
                        ${_LIBRARY})
endif()

add_timemory_google_test(apply_tests
    DISCOVER_TESTS
    SOURCES         apply_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

if(TIMEMORY_USE_ARCH AND OFF)
    add_timemory_google_test(aligned_allocator_tests
        DISCOVER_TESTS
        SOURCES         aligned_allocator_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch timemory-compile-options
                        timemory-develop-options timemory-plotting timemory-analysis-tools
                        ${_LIBRARY})
endif()

set_source_files_properties(cuda_tests.cpp cupti_tests.cpp cupti_profiler_tests.cpp
    PROPERTIES
    LANGUAGE        ${DEVICE_LANGUAGE}
    LINKER_LANGUAGE ${DEVICE_LANGUAGE})

if(TIMEMORY_USE_CUDA)
    add_timemory_google_test(cuda_tests
        DISCOVER_TESTS
        SOURCES         cuda_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options
                        timemory-cuda    timemory-cudart timemory-develop-options
                        timemory-plotting timemory-analysis-tools
                        ${_LIBRARY})
endif()

if(TIMEMORY_USE_CUDA AND TIMEMORY_USE_NVTX)
    add_timemory_google_test(cuda_nvtx_tests
        DISCOVER_TESTS
        SOURCES         cuda_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options timemory-develop-options
                        timemory-cuda    timemory-cudart timemory-nvtx
                        timemory-plotting timemory-analysis-tools
                        ${_LIBRARY})
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_DRIVER_LIBRARY)
    add_library(cupti-compile-options INTERFACE)
    target_compile_options(cupti-compile-options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

    add_timemory_google_test(cupti_tests
        DISCOVER_TESTS
        SOURCES         cupti_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options timemory-develop-options
                        timemory-cuda    timemory-cupti  # timemory-cudart-device
                        cupti-compile-options timemory-plotting timemory-analysis-tools
                        timemory-arch)

    # NVPERF library
    set(nvperf_valid_archs volta turing)
    if(CUPTI_nvperf_host_LIBRARY AND "${TIMEMORY_CUDA_ARCH}" IN_LIST nvperf_valid_archs)
        add_timemory_google_test(cupti_profiler_tests
            DISCOVER_TESTS
            SOURCES         cupti_profiler_tests.cpp
            LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options timemory-develop-options
                            timemory-cuda    timemory-cupti  timemory-cudart-device
                            cupti-compile-options timemory-plotting timemory-analysis-tools
                            timemory-arch)
    endif()
endif()

if(_HAS_PROFILER)
    add_timemory_google_test(gperf_cpu_tests
        SOURCES         gperf_cpu_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                        timemory-gperftools-cpu timemory-plotting timemory-analysis-tools
                        ${_LIBRARY}
        ENVIRONMENT     "CPUPROFILE_FREQUENCY=1000;CPUPROFILE_REALTIME=1")
endif()

add_timemory_google_test(gperf_fake_cpu_tests
    DISCOVER_TESTS
    SOURCES         gperf_cpu_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

if(_HAS_TCMALLOC)
    add_timemory_google_test(gperf_heap_tests
        DISCOVER_TESTS
        SOURCES         gperf_heap_tests.cpp
        LINK_LIBRARIES  timemory-headers    timemory-arch   timemory-compile-options timemory-develop-options
                        timemory-gperftools-heap timemory-plotting timemory-analysis-tools
                        ${_LIBRARY})
endif()

if(TIMEMORY_USE_GOTCHA)
    add_timemory_google_test(gotcha_tests
        DISCOVER_TESTS
        SOURCES         gotcha_tests.cpp
        LINK_LIBRARIES  timemory-headers    timemory-compile-options timemory-develop-options
                        timemory-gotcha     timemory-mpi
                        timemory-papi       timemory-vector
                        timemory-arch       gotcha-tests-lib
                        timemory-plotting   timemory-analysis-tools
                        ${_LIBRARY})
endif()

add_timemory_google_test(priority_tests
    DISCOVER_TESTS
    SOURCES         priority_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools
                    ${_LIBRARY})

if(TIMEMORY_USE_MPI)
    add_timemory_google_test(mpi_tests
        DISCOVER_TESTS
        SOURCES         mpi_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                        timemory-plotting timemory-analysis-tools timemory-mpi
                        ${_LIBRARY})
endif()

if(TIMEMORY_USE_UPCXX)
    add_timemory_google_test(upcxx_tests
        DISCOVER_TESTS
        SOURCES         upcxx_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                        timemory-plotting timemory-analysis-tools timemory-upcxx
                        ${_LIBRARY})
endif()

add_timemory_google_test(variadic_tests
    DISCOVER_TESTS
    SOURCES         variadic_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools timemory-mpi
                    ${_LIBRARY})

add_timemory_google_test(derived_tests
    DISCOVER_TESTS
    SOURCES         derived_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options timemory-develop-options
                    timemory-plotting timemory-analysis-tools timemory-mpi
                    ${_LIBRARY})

set(_OMPT_TARGET)
if(TARGET timemory-ompt-shared)
    set(_OMPT_TARGET timemory-ompt-shared)
elseif(TARGET timemory-ompt-static)
    set(_OMPT_TARGET timemory-ompt-static)
endif()

if(_OMPT_TARGET)
    add_timemory_google_test(ompt_handle_tests
        DISCOVER_TESTS
        SOURCES         ompt_handle_tests.cpp
        LINK_LIBRARIES  ${_OMPT_TARGET})
endif()
