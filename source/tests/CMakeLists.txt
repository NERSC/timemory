
# include the current directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_UNITY_BUILD OFF)

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

if(TARGET timemory::timemory-cxx-shared)
    set(_LIBRARY_TARGET timemory::timemory-cxx-shared)
elseif(TARGET timemory::timemory-cxx-static)
    set(_LIBRARY_TARGET timemory::timemory-cxx-static)
endif()

set(_LIBRARY)
if(TIMEMORY_USE_TESTING_LINK_LIBRARY)
    set(_LIBRARY ${_LIBRARY_TARGET})
endif()

if(TIMEMORY_USE_OMPT)
    find_package(OpenMP QUIET)
    if(OpenMP_FOUND)
        set(_OPENMP OpenMP::OpenMP_CXX)
    endif()
endif()

# Optimization flags
add_library(test-opt-flags INTERFACE)
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
target_compile_options(test-opt-flags INTERFACE ${_FLAGS})
if(NOT TIMEMORY_USE_ARCH)
  add_target_flag_if_avail(test-opt-flags "-march=native")
endif()

# Debug flags
add_library(test-debug-flags INTERFACE)
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
target_compile_options(test-debug-flags INTERFACE ${_FLAGS})
add_target_flag_if_avail(test-debug-flags "-O0" "-g")

# Any warnings trigger an error flags
add_library(test-werror-flags INTERFACE)
add_target_flag_if_avail(test-werror-flags "-W" "-Wall" "-Wextra" "-pedantic"
    "-Wno-mismatched-tags" "-Werror")
target_link_libraries(test-werror-flags INTERFACE
    timemory::timemory-headers timemory::timemory-compile-timing)
target_compile_definitions(test-werror-flags INTERFACE TIMEMORY_CMAKE)

add_library(common-test-libs INTERFACE)
target_link_libraries(common-test-libs INTERFACE
    timemory::timemory-headers
    timemory::timemory-compile-options
    timemory::timemory-develop-options
    timemory::timemory-analysis-tools)

set(HYBRID_COMPONENTS )

if(TIMEMORY_USE_CUDA)
    list(APPEND HYBRID_COMPONENTS
        timemory::timemory-cuda
        timemory::timemory-cudart)
endif()

if(TIMEMORY_USE_PAPI)
    list(APPEND HYBRID_COMPONENTS
        timemory::timemory-papi)
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_DRIVER_LIBRARY)
    list(APPEND HYBRID_COMPONENTS
        timemory::timemory-cupti
        timemory::timemory-cudart-device)
endif()

if(TIMEMORY_USE_CALIPER)
    list(APPEND HYBRID_COMPONENTS
        timemory::timemory-caliper)
endif()

if(TIMEMORY_USE_VTUNE)
    list(APPEND HYBRID_COMPONENTS
        timemory::timemory-vtune)
endif()

set(ENV{TIMEMORY_ENABLED} OFF)

add_subdirectory(external)

if(TARGET custom-record-functions)
    add_timemory_google_test(library_tests
        DISCOVER_TESTS
        SOURCES         library_tests.cpp
        LINK_LIBRARIES  custom-record-functions
                        timemory::timemory-core)
endif()

if(_LIBRARY_TARGET)
    list(APPEND trace_tests_env "TIMEMORY_COLLAPSE_THREADS=ON")
    list(APPEND trace_tests_env "TIMEMORY_NODE_COUNT=2")
    list(APPEND trace_tests_env "TIMEMORY_MPI_THREAD=ON")
    list(APPEND trace_tests_env "TIMEMORY_MPI_THREAD_TYPE=funneled")

    add_timemory_google_test(trace_tests
        MPI
        NPROC           4
        SOURCES         trace_tests.cpp
        LINK_LIBRARIES  ${_LIBRARY_TARGET}
                        timemory::timemory-dmp
                        timemory::timemory-plotting
                        timemory::timemory-mpip-library
                        timemory::timemory-ompt-library
        ENVIRONMENT     ${trace_tests_env})

    add_timemory_google_test(throttle_tests
        DISCOVER_TESTS
        SOURCES         throttle_tests.cpp
        LINK_LIBRARIES  ${_LIBRARY_TARGET})
endif()

add_timemory_google_test(api_tests
    DISCOVER_TESTS
    SOURCES         api_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core
                    timemory::timemory-timing-component)

add_timemory_google_test(type_trait_tests
    DISCOVER_TESTS
    SOURCES         type_trait_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core)

add_timemory_google_test(warning_tests
    SOURCES         warning_tests.cpp
    LINK_LIBRARIES  test-werror-flags
                    test-debug-flags
                    timemory::timemory-core)

add_timemory_google_test(socket_tests
    SOURCES         socket_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core
                    test-werror-flags)

add_timemory_google_test(settings_tests
    DISCOVER_TESTS
    SOURCES         settings_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core
                    timemory::timemory-config)

add_timemory_google_test(argparse_tests
    SOURCES         argparse_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core)

list(APPEND component_bundle_tests_env "TIMEMORY_COLLAPSE_PROCESSES=OFF")
list(APPEND component_bundle_tests_env "TIMEMORY_COLLAPSE_THREADS=OFF")
list(APPEND component_bundle_tests_env "TIMEMORY_MPI_THREAD=OFF")

add_timemory_google_test(component_bundle_tests
    MPI
    NPROCS          2
    SOURCES         component_bundle_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core
    ENVIRONMENT     ${component_bundle_tests_env})

list(APPEND kokkosp_tests_env "TIMEMORY_COLLAPSE_THREADS=ON")
list(APPEND kokkosp_tests_env "TIMEMORY_NODE_COUNT=1")
list(APPEND kokkosp_tests_env "TIMEMORY_MPI_THREAD=OFF")

add_timemory_google_test(kokkosp_tests
    MPI
    NPROCS          2
    SOURCES         kokkosp_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core
                    ${_LIBRARY}
    ENVIRONMENT     ${kokkosp_tests_env})

add_timemory_google_test(flat_tests
    DISCOVER_TESTS
    SOURCES         flat_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    extern-test-templates)

add_timemory_google_test(timeline_tests
    DISCOVER_TESTS
    SOURCES         timeline_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    extern-test-templates)

add_timemory_google_test(data_tracker_tests
    DISCOVER_TESTS
    SOURCES         data_tracker_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core)

add_timemory_google_test(cache_tests
    SOURCES         cache_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-core
                    timemory::timemory-timing-component
                    timemory::timemory-rusage-component)

add_timemory_google_test(archive_storage_tests
    SOURCES         archive_storage_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-mpi
                    timemory::timemory-upcxx
                    timemory::timemory-core
                    timemory::timemory-timing-component)

add_timemory_google_test(threading_tests
    DISCOVER_TESTS
    SOURCES         threading_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    timemory::timemory-ompt
                    ${_LIBRARY} ${_OPENMP})

add_timemory_google_test(stl_overload_tests
    DISCOVER_TESTS
    SOURCES         stl_overload_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

add_timemory_google_test(user_bundle_tests
    DISCOVER_TESTS
    SOURCES         user_bundle_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

add_timemory_google_test(mangle_tests
    DISCOVER_TESTS
    SOURCES         mangle_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

add_timemory_google_test(tuple_tests
    DISCOVER_TESTS
    SOURCES         tuple_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-papi
                    timemory::timemory-plotting
                    timemory::timemory-caliper
                    ${_LIBRARY})

add_timemory_google_test(macro_tests
    DISCOVER_TESTS
    SOURCES         macro_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

add_timemory_google_test(hybrid_tests
    DISCOVER_TESTS
    SOURCES         hybrid_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    ${HYBRID_COMPONENTS}
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

add_timemory_google_test(timing_tests
    DISCOVER_TESTS
    SOURCES         timing_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

add_timemory_google_test(rusage_tests
    DISCOVER_TESTS
    SOURCES         rusage_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

if(TIMEMORY_USE_PAPI)
    add_timemory_google_test(papi_tests
        DISCOVER_TESTS
        SOURCES         papi_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-papi
                        timemory::timemory-plotting
                        timemory::timemory-arch
                        timemory::timemory-roofline-options
                        test-opt-flags
                        ${_LIBRARY})

    # roofline
    add_timemory_google_test(cpu_roofline_tests
        DISCOVER_TESTS
        SOURCES         cpu_roofline_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-papi
                        timemory::timemory-plotting
                        timemory::timemory-arch
                        ${_LIBRARY})
endif()

add_timemory_google_test(apply_tests
    DISCOVER_TESTS
    SOURCES         apply_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

if(TIMEMORY_USE_ARCH)
    add_timemory_google_test(aligned_allocator_tests
        DISCOVER_TESTS
        SOURCES         aligned_allocator_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-arch
                        timemory::timemory-core
                        timemory::timemory-plotting
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
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-arch
                        timemory::timemory-core
                        timemory::timemory-cuda
                        timemory::timemory-cudart
                        timemory::timemory-plotting
                        ${_LIBRARY})
    #
    add_timemory_google_test(cuda_nvtx_tests
        DISCOVER_TESTS
        SOURCES         cuda_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-arch
                        timemory::timemory-core
                        timemory::timemory-cuda
                        timemory::timemory-cudart
                        timemory::timemory-plotting
                        ${_LIBRARY})
endif()

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_DRIVER_LIBRARY)
    add_library(cupti-compile-options INTERFACE)
    target_compile_options(cupti-compile-options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)

    add_timemory_google_test(cupti_tests
        DISCOVER_TESTS
        SOURCES         cupti_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-arch
                        timemory::timemory-core
                        timemory::timemory-cuda
                        timemory::timemory-cupti
                        timemory::timemory-plotting
                        # timemory::timemory-cudart-device
                        cupti-compile-options)

    # NVPERF library
    set(nvperf_valid_archs volta turing)
    if(CUPTI_nvperf_host_LIBRARY AND "${TIMEMORY_CUDA_ARCH}" IN_LIST nvperf_valid_archs)
        add_timemory_google_test(cupti_profiler_tests
            DISCOVER_TESTS
            SOURCES         cupti_profiler_tests.cpp
            LINK_LIBRARIES  common-test-libs
                            timemory::timemory-arch
                            timemory::timemory-core
                            timemory::timemory-cuda
                            timemory::timemory-cupti
                            timemory::timemory-plotting
                            timemory::timemory-cudart-device
                            cupti-compile-options)
    endif()
endif()

if(TIMEMORY_USE_GPERFTOOLS)
    add_timemory_google_test(gperf_cpu_tests
        SOURCES         gperf_cpu_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-gperftools
                        timemory::timemory-plotting
                        timemory::timemory-core
                        ${_LIBRARY}
        ENVIRONMENT     "CPUPROFILE_FREQUENCY=1000;CPUPROFILE_REALTIME=1")
    add_timemory_google_test(gperf_heap_tests
        DISCOVER_TESTS
        SOURCES         gperf_heap_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-arch
                        timemory::timemory-gperftools
                        timemory::timemory-plotting
                        timemory::timemory-core
                        ${_LIBRARY})
endif()

add_timemory_google_test(gperf_fake_cpu_tests
    DISCOVER_TESTS
    SOURCES         gperf_cpu_tests.cpp
    LINK_LIBRARIES  timemory::timemory-headers
                    timemory::timemory-plotting
                    timemory::timemory-analysis-tools
                    timemory::timemory-core
                    ${_LIBRARY})


if(TIMEMORY_USE_GOTCHA)
    add_timemory_google_test(gotcha_tests
        DISCOVER_TESTS
        SOURCES         gotcha_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        gotcha-tests-lib
                        timemory::timemory-gotcha
                        timemory::timemory-dmp
                        timemory::timemory-vector
                        timemory::timemory-arch
                        timemory::timemory-plotting
                        timemory::timemory-core
                        ${_LIBRARY})
endif()

add_timemory_google_test(priority_tests
    DISCOVER_TESTS
    SOURCES         priority_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    ${_LIBRARY})

if(TIMEMORY_USE_MPI)
    #
    list(APPEND mpi_tests_env "TIMEMORY_MPI_THREAD_TYPE=serialized")
    list(APPEND mpi_tests_env "TIMEMORY_SEPARATOR_FREQ=0")
    #
    add_timemory_google_test(mpi_tests
        MPI
        NPROC           2
        SOURCES         mpi_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-plotting
                        timemory::timemory-dmp
                        timemory::timemory-core
                        ${_LIBRARY}
        ENVIRONMENT     "${mpi_tests_env}")
    #
    list(APPEND mpi_tests_env "TIMEMORY_DIFF_OUTPUT=ON")
    list(APPEND mpi_tests_env "TIMEMORY_OUTPUT_PATH=timemory-mpi-tests-diff-output")
    list(APPEND mpi_tests_env
        "TIMEMORY_INPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/timemory-mpi-tests-output")
    #
    add_timemory_google_test(mpi_tests_diff
        MPI
        NPROC           2
        DEPENDS         mpi_tests
        COMMAND         $<TARGET_FILE:mpi_tests>
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-plotting
                        timemory::timemory-dmp
                        timemory::timemory-core
                        ${_LIBRARY}
        ENVIRONMENT     "${mpi_tests_env}")

endif()

if(TIMEMORY_USE_UPCXX)
    add_timemory_google_test(upcxx_tests
        DISCOVER_TESTS
        SOURCES         upcxx_tests.cpp
        LINK_LIBRARIES  common-test-libs
                        timemory::timemory-plotting
                        timemory::timemory-upcxx
                        timemory::timemory-core
                        ${_LIBRARY})
endif()

add_timemory_google_test(variadic_tests
    DISCOVER_TESTS
    SOURCES         variadic_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    timemory::timemory-dmp
                    ${_LIBRARY})

add_timemory_google_test(derived_tests
    DISCOVER_TESTS
    SOURCES         derived_tests.cpp
    LINK_LIBRARIES  common-test-libs
                    timemory::timemory-plotting
                    timemory::timemory-core
                    timemory::timemory-dmp
                    ${_LIBRARY})

set(_OMPT_TARGET)
if(TARGET timemory::timemory-ompt-shared)
    set(_OMPT_TARGET timemory::timemory-ompt-shared)
elseif(TARGET timemory::timemory-ompt-static)
    set(_OMPT_TARGET timemory::timemory-ompt-static)
endif()

if(_OMPT_TARGET AND _OPENMP)
    add_timemory_google_test(ompt_handle_tests
        DISCOVER_TESTS
        SOURCES         ompt_handle_tests.cpp
        LINK_LIBRARIES  ${_OMPT_TARGET} ${_OPENMP})
endif()

if(TARGET timemory::timemory-ncclp-library AND TARGET nccl_test_interface)
    add_timemory_google_test(nccl_tests
        SOURCES         nccl_tests.cpp
        LINK_LIBRARIES  nccl_test_interface
                        timemory::timemory-ncclp-library)
endif()
