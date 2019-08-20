
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

add_googletest(macro_tests
    DISCOVER_TESTS
    SOURCES         macro_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options)

add_googletest(hybrid_tests
    DISCOVER_TESTS
    SOURCES         hybrid_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options ${HYBRID_COMPONENTS})

add_googletest(timing_tests
    DISCOVER_TESTS
    SOURCES         timing_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options)

add_googletest(rusage_tests
    DISCOVER_TESTS
    SOURCES         rusage_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options)

add_googletest(papi_tests
    DISCOVER_TESTS
    SOURCES         papi_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-papi timemory-compile-options)

add_googletest(apply_tests
    DISCOVER_TESTS
    SOURCES         apply_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-compile-options)

if(TIMEMORY_USE_ARCH)
    add_googletest(aligned_allocator_tests
        DISCOVER_TESTS
        SOURCES         aligned_allocator_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch timemory-compile-options)
endif()

set_source_files_properties(cuda_tests.cpp cupti_tests.cpp
    PROPERTIES
    LANGUAGE        ${DEVICE_LANGUAGE}
    LINKER_LANGUAGE ${DEVICE_LANGUAGE})

add_googletest(cuda_tests
    DISCOVER_TESTS
    SOURCES         cuda_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options
                    timemory-cuda    timemory-cudart)

add_googletest(cuda_nvtx_tests
    DISCOVER_TESTS
    SOURCES         cuda_tests.cpp
    LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options
                    timemory-cuda    timemory-cudart timemory-cuda-nvtx)

if(TIMEMORY_USE_CUPTI AND HAS_CUDA_cuda_LIBRARY)
    add_googletest(cupti_tests
        DISCOVER_TESTS
        SOURCES         cupti_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options
                        timemory-cuda    timemory-cupti  timemory-cudart-device)
endif()
