
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY )
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY )
unset(CMAKE_PDB_OUTPUT_DIRECTORY )

add_library(preload-stubs-shared SHARED stubs.cpp)

add_googletest(preload_tests
    DISCOVER_TESTS
    SOURCES         preload_tests.cpp
    LINK_LIBRARIES  preload-stubs-shared
)
