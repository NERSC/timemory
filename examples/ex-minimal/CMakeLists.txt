cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(timemory-Minimal-Example LANGUAGES C CXX)

set(timemory_FIND_COMPONENTS_INTERFACE timemory-minimal-example)
set(COMPONENTS headers compile-options analysis-tools OPTIONAL_COMPONENTS cxx)

option(USE_PAPI "Enable PAPI support" OFF)
option(USE_CALIPER "Enable Caliper support" OFF)
if(USE_PAPI)
  list(APPEND COMPONENTS papi)
endif()
if(USE_CALIPER)
  list(APPEND COMPONENTS caliper)
endif()
find_package(timemory REQUIRED COMPONENTS ${COMPONENTS})

set(TIMEMORY_C_LIBRARY )
if("C" IN_LIST timemory_ENABLED_LINK_LANGUAGES)
    if("SHARED" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_C_LIBRARY timemory-c-shared)
    elseif("STATIC" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_C_LIBRARY timemory-c-static)
    endif()
endif()

set(TIMEMORY_CXX_LIBRARY )
if("CXX" IN_LIST timemory_ENABLED_LINK_LANGUAGES)
    if("SHARED" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_CXX_LIBRARY timemory-cxx-shared)
    elseif("STATIC" IN_LIST timemory_ENABLED_LINK_LIBRARIES)
        set(TIMEMORY_CXX_LIBRARY timemory-cxx-static)
    endif()
endif()

# if C library is available
if(TIMEMORY_C_LIBRARY)
    add_executable(ex_c_minimal ex_minimal.c)
    target_link_libraries(ex_c_minimal timemory-minimal-example ${TIMEMORY_C_LIBRARY})
    install(TARGETS ex_c_minimal DESTINATION bin)

    # if C stubs via CXX library are available
    if(TIMEMORY_CXX_LIBRARY)
        add_executable(ex_c_minimal_library_overload ex_minimal_library_overload.c)
        target_link_libraries(ex_c_minimal_library_overload timemory-minimal-example
            ${TIMEMORY_C_LIBRARY} ${TIMEMORY_CXX_LIBRARY})
        install(TARGETS ex_c_minimal_library_overload DESTINATION bin)
    endif()
endif()

# if C stubs via CXX library are available
if(TIMEMORY_CXX_LIBRARY)
    add_executable(ex_c_minimal_library ex_minimal_library.c)
    target_link_libraries(ex_c_minimal_library timemory-minimal-example
        ${TIMEMORY_CXX_LIBRARY} ${TIMEMORY_C_LIBRARY})
    install(TARGETS ex_c_minimal_library DESTINATION bin)
endif()

# headers are always available
add_executable(ex_cxx_minimal ex_minimal.cpp)
target_link_libraries(ex_cxx_minimal timemory-minimal-example)
install(TARGETS ex_cxx_minimal DESTINATION bin)

if(TIMEMORY_CXX_LIBRARY)
    add_executable(ex_cxx_minimal_library ex_minimal_library.cpp)
    target_link_libraries(ex_cxx_minimal_library timemory-minimal-example ${TIMEMORY_CXX_LIBRARY})
    install(TARGETS ex_cxx_minimal_library DESTINATION bin)

    add_executable(ex_cxx_minimal_library_overload ex_minimal_library_overload.cpp)
    target_link_libraries(ex_cxx_minimal_library_overload timemory-minimal-example ${TIMEMORY_CXX_LIBRARY})
    install(TARGETS ex_cxx_minimal_library_overload DESTINATION bin)
endif()

if(NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE "/usr/bin/env python")
endif()

configure_file(${PROJECT_SOURCE_DIR}/ex_python_minimal.py
    ${CMAKE_BINARY_DIR}/ex_python_minimal @ONLY)

install(
    FILES ${CMAKE_BINARY_DIR}/ex_python_minimal
    DESTINATION bin
    PERMISSIONS
        OWNER_EXECUTE OWNER_READ OWNER_WRITE
        GROUP_EXECUTE GROUP_READ
        WORLD_EXECUTE WORLD_READ)
