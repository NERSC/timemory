
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY )
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY )
unset(CMAKE_PDB_OUTPUT_DIRECTORY )

add_googletest(timing_tests
    DISCOVER_TESTS
    SOURCES         timing_tests.cpp
    LINK_LIBRARIES  timemory-headers
)

add_googletest(rusage_tests
    DISCOVER_TESTS
    SOURCES         rusage_tests.cpp
    LINK_LIBRARIES  timemory-headers
)
