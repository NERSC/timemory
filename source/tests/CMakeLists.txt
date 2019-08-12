
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY )
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY )
unset(CMAKE_PDB_OUTPUT_DIRECTORY )

set(DEVICE_LANGUAGE CXX)
if(TIMEMORY_USE_CUDA)
    set(DEVICE_LANGUAGE CUDA)
endif()

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
    LINK_LIBRARIES  timemory-headers timemory-arch  timemory-compile-options
                    timemory-cuda    timemory-cudart)

if(TIMEMORY_USE_CUPTI)
    add_googletest(cupti_tests
        DISCOVER_TESTS
        SOURCES         cupti_tests.cpp
        LINK_LIBRARIES  timemory-headers timemory-arch   timemory-compile-options
                        timemory-cuda    timemory-cupti  timemory-cudart-device
    )
endif()
