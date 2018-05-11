
## -- CTest Config
if(EXISTS "${CMAKE_SOURCE_DIR}/CTestConfig.cmake")
    configure_file(${CMAKE_SOURCE_DIR}/CTestConfig.cmake
        ${CMAKE_BINARY_DIR}/CTestConfig.cmake @ONLY)
endif(EXISTS "${CMAKE_SOURCE_DIR}/CTestConfig.cmake")

# testing
ENABLE_TESTING()
if(TIMEMORY_BUILD_TESTING)
    include(CTest)
endif(TIMEMORY_BUILD_TESTING)


# ------------------------------------------------------------------------ #
# -- Function to create a temporary directory
# ------------------------------------------------------------------------ #
function(GET_TEMPORARY_DIRECTORY DIR_VAR DIR_MODEL)
    # create a root working directory
    if(WIN32)
        set(_TMP_DIR "$ENV{TEMP}")
        STRING(REPLACE "\\" "/" _TMP_DIR "${_TMP_DIR}")
        set(_TMP_ROOT "${_TMP_DIR}/${PROJECT_NAME}/cdash/${DIR_MODEL}")
    else(WIN32)
        set(_TMP_ROOT "/tmp/${PROJECT_NAME}/cdash/${DIR_MODEL}")
    endif(WIN32)
    set(${DIR_VAR} "${_TMP_ROOT}" PARENT_SCOPE)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${_TMP_ROOT}) 
endfunction()


# ------------------------------------------------------------------------ #
# -- Configure Branch label
# ------------------------------------------------------------------------ #
if(TIMEMORY_BUILD_TESTING)

    find_package(Git REQUIRED)

    execute_process(COMMAND ${GIT_EXECUTABLE} name-rev --name-only HEAD
        OUTPUT_VARIABLE CMAKE_SOURCE_BRANCH
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "~[0-9]+" "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")
    string(REGEX REPLACE "tags/" "" CMAKE_SOURCE_BRANCH "${CMAKE_SOURCE_BRANCH}")

    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --verify HEAD
        OUTPUT_VARIABLE CMAKE_SOURCE_REVISION
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE)

endif(TIMEMORY_BUILD_TESTING)


# ------------------------------------------------------------------------ #
# -- Add options
# ------------------------------------------------------------------------ #
macro(add_ctest_options VARIABLE )

    get_cmake_property(_vars CACHE_VARIABLES)
    get_cmake_property(_nvars VARIABLES)
    foreach(_var ${_nvars})
        list(APPEND _vars ${_var})
    endforeach(_var ${_nvars})

    list(REMOVE_DUPLICATES _vars)
    list(SORT _vars)
    set(_set_vars ${ARGN})
    foreach(_var ${_vars})
        STRING(REGEX MATCH "^TIMEMORY_USE_" _use_found "${_var}")
        STRING(REGEX MATCH ".*(_ROOT|_LIBRARY|_INCLUDE_DIR|_EXECUTABLE)$"
            _root_found "${_var}")
        STRING(REGEX MATCH "^(PREVIOUS_|CMAKE_|OSX_|DEFAULT_|EXTERNAL_|_|CTEST_|DOXYGEN_|QT_)"
            _skip_prefix "${_var}")
        STRING(REGEX MATCH ".*(_AVAILABLE|_LIBRARIES|_INCLUDE_DIRS)$"
            _skip_suffix "${_var}")

        if(_skip_prefix OR _skip_suffix)
            continue()
        endif(_skip_prefix OR _skip_suffix)

        if(_use_found OR _root_found)
            list(APPEND _set_vars ${_var})
        endif(_use_found OR _root_found)
    endforeach(_var ${_vars})

    list(REMOVE_DUPLICATES _set_vars)
    list(SORT _set_vars)
    foreach(_var ${_set_vars})
        if(NOT "${${_var}}" STREQUAL "" AND
            NOT "${${_var}}" STREQUAL "${_var}-NOTFOUND")
            add(${VARIABLE} "-D${_var}='${${_var}}'")
        endif(NOT "${${_var}}" STREQUAL "" AND
            NOT "${${_var}}" STREQUAL "${_var}-NOTFOUND")
    endforeach(_var ${_set_vars})

    unset(_vars)
    unset(_nvars)
    unset(_set_vars)

endmacro(add_ctest_options VARIABLE )


# ------------------------------------------------------------------------ #
# -- Configure CTest for CDash
# ------------------------------------------------------------------------ #
if(NOT TIMEMORY_DASHBOARD_MODE AND TIMEMORY_BUILD_TESTING)

    # get temporary directory for dashboard testing
    if(NOT DEFINED CMAKE_DASHBOARD_ROOT)
        GET_TEMPORARY_DIRECTORY(CMAKE_DASHBOARD_ROOT ${CTEST_MODEL})
    endif(NOT DEFINED CMAKE_DASHBOARD_ROOT)
    
    # set the CMake configure options
    add_ctest_options(CMAKE_CONFIGURE_OPTIONS
        CMAKE_BUILD_TYPE
        CMAKE_C_COMPILER CMAKE_CXX_COMPILER
        MPI_C_COMPILER MPI_CXX_COMPILER
        CTEST_MODEL CTEST_SITE
        TIMEMORY_EXCEPTIONS
        BUILD_SHARED_LIBS
        TIMEMORY_BUILD_EXAMPLES
        TIMEMORY_TEST_MPI)

    set(cdash_templates Init Build Test Submit Glob Stages)
    if(USE_COVERAGE)
        list(APPEND cdash_templates Coverage)
    endif(USE_COVERAGE)
    if(MEMORYCHECK_COMMAND)
        list(APPEND cdash_templates MemCheck)
    endif(MEMORYCHECK_COMMAND)

    foreach(_type ${cdash_templates})
        ## -- CTest Setup
        if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Templates/cdash/${_type}.cmake.in")
            configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/cdash/${_type}.cmake.in
                ${CMAKE_BINARY_DIR}/cdash/${_type}.cmake @ONLY)
        endif(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Templates/cdash/${_type}.cmake.in")
    endforeach(_type Init Build Test Coverage MemCheck Submit Glob Stages)

    ## -- CTest Custom
    if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Templates/CTestCustom.cmake.in")
        configure_file(${CMAKE_SOURCE_DIR}/cmake/Templates/CTestCustom.cmake.in
            ${CMAKE_BINARY_DIR}/CTestCustom.cmake @ONLY)
    endif(EXISTS "${CMAKE_SOURCE_DIR}/cmake/Templates/CTestCustom.cmake.in")

endif(NOT TIMEMORY_DASHBOARD_MODE AND TIMEMORY_BUILD_TESTING)


# ---------------------------------------------------------------------------- #
# -- Add tests
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#   Python tests
#
if(TIMEMORY_USE_PYTHON_BINDING)

    add_test(NAME python_simple
        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/simple_test.py
            --enable-dart --write-ctest-notes
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(python_simple PROPERTIES
        LABELS "python;unit_test" TIMEOUT 7200)

    add_test(NAME python_unit_test
        # we don't enable dart because this always print
        # (unittest will repeat DartMeasurementFiles)
        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/timemory_test.py
            --write-ctest-notes
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(python_unit_test PROPERTIES
        LABELS "python;unit_test" TIMEOUT 7200)

    add_test(NAME python_nested
        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/nested_test.py
            --enable-dart --write-ctest-notes
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(python_nested PROPERTIES
        LABELS "python;unit_test" TIMEOUT 7200)

    add_test(NAME python_array
        COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_BINARY_DIR}/array_test.py
            --enable-dart --write-ctest-notes
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(python_array PROPERTIES
        LABELS "python;unit_test" TIMEOUT 7200)

endif(TIMEMORY_USE_PYTHON_BINDING)

# ---------------------------------------------------------------------------- #
#   Compiled tests
#
if(TIMEMORY_BUILD_EXAMPLES)

    if(TIMEMORY_USE_MPI AND MPI_FOUND AND TIMEMORY_TEST_MPI)
        set(_TEST_MPI ON)
    else(TIMEMORY_USE_MPI AND MPI_FOUND AND TIMEMORY_TEST_MPI)
        set(_TEST_MPI OFF)
    endif(TIMEMORY_USE_MPI AND MPI_FOUND AND TIMEMORY_TEST_MPI)

    #----------------------------------------------#
    #   C Timing
    #
    set(TEST_NAME test_c_timing)
    add_test(NAME ${TEST_NAME}
        COMMAND $<TARGET_FILE:${TEST_NAME}>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "c;unit_test" TIMEOUT 7200)

    #----------------------------------------------#
    #   CXX Timing
    #
    set(TEST_NAME test_cxx_timing)
    add_test(NAME ${TEST_NAME}
        COMMAND $<TARGET_FILE:${TEST_NAME}>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "cxx;unit_test" TIMEOUT 7200)

    #----------------------------------------------#
    #   CXX Overhead
    #
    set(TEST_NAME test_cxx_overhead)
    add_test(NAME ${TEST_NAME}
        COMMAND $<TARGET_FILE:${TEST_NAME}>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "cxx;unit_test" TIMEOUT 7200)

    #----------------------------------------------#
    #   CXX Total
    #
    set(TEST_NAME test_cxx_total)
    add_test(NAME ${TEST_NAME}
        COMMAND $<TARGET_FILE:${TEST_NAME}>
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(${TEST_NAME} PROPERTIES
        LABELS "cxx;unit_test" TIMEOUT 7200)

    #----------------------------------------------#
    #   CXX + MPI Timing
    #
    if(_TEST_MPI)
        set(TEST_NAME test_cxx_mpi_timing)
        add_test(NAME ${TEST_NAME}
            COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:${TEST_NAME}>
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        set_tests_properties(${TEST_NAME} PROPERTIES
            LABELS "cxx;unit_test;mpi" TIMEOUT 7200)
    endif(_TEST_MPI)

    unset(_TEST_MPI)

endif(TIMEMORY_BUILD_EXAMPLES)
