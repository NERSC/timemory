
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY )
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY )
unset(CMAKE_PDB_OUTPUT_DIRECTORY )

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
