# include guard
include_guard(DIRECTORY)

# MacroUtilities - useful macros and functions for generic tasks
#

cmake_policy(PUSH)
cmake_policy(SET CMP0054 NEW)

include(CMakeDependentOption)
include(CMakeParseArguments)

# Make this file generic by not explicitly defining the name of the project
string(TOUPPER "${PROJECT_NAME}" PROJECT_NAME_UC)
string(TOLOWER "${PROJECT_NAME}" PROJECT_NAME_LC)

unset(${PROJECT_NAME_UC}_COMPILED_LIBRARIES CACHE)
unset(${PROJECT_NAME_UC}_INTERFACE_LIBRARIES CACHE)

#-----------------------------------------------------------------------
# CACHED LIST
#-----------------------------------------------------------------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
MACRO(CACHE_LIST _OP _LIST)
    set(_TMP_CACHE_LIST ${${_LIST}})
    # apply operation on list
    list(${_OP} _TMP_CACHE_LIST ${ARGN})
    # replace list
    # set(${_LIST} ${_TMP_CACHE_LIST})
    # if(NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    #     set(${_LIST} ${_TMP_CACHE_LIST} PARENT_SCOPE)
    # endif()
    # apply operation on list
    #list(${_OP} ${_LIST} ${ARGN})
    # replace list
    set(${_LIST} "${_TMP_CACHE_LIST}" CACHE INTERNAL "" FORCE)
ENDMACRO()


#-----------------------------------------------------------------------
# CMAKE EXTENSIONS
#-----------------------------------------------------------------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
MACRO(SET_IFNOT _var _value)
    if(NOT DEFINED ${_var})
        set(${_var} ${_value} ${ARGN})
    endif()
ENDMACRO()


#-----------------------------------------------------------------------
# macro safe_remove_duplicates(<list>)
#       ensures remove_duplicates is only called if list has values
#
MACRO(SAFE_REMOVE_DUPLICATES _list)
    if(NOT "${${_list}}" STREQUAL "")
        list(REMOVE_DUPLICATES ${_list})
    endif(NOT "${${_list}}" STREQUAL "")
ENDMACRO()


#-----------------------------------------------------------------------
# function - capitalize - make a string capitalized (first letter is capital)
#   usage:
#       capitalize("SHARED" CShared)
#   message(STATUS "-- CShared is \"${CShared}\"")
#   $ -- CShared is "Shared"
FUNCTION(CAPITALIZE str var)
    # make string lower
    string(TOLOWER "${str}" str)
    string(SUBSTRING "${str}" 0 1 _first)
    string(TOUPPER "${_first}" _first)
    string(SUBSTRING "${str}" 1 -1 _remainder)
    string(CONCAT str "${_first}" "${_remainder}")
    set(${var} "${str}" PARENT_SCOPE)
ENDFUNCTION()


#-----------------------------------------------------------------------
# macro set_ifnot_match(<var> <value>)
#       If variable var is not set, set its value to that provided
#
MACRO(SET_IFNOT_MATCH VAR APPEND)
    if(NOT "${APPEND}" STREQUAL "")
        STRING(REGEX MATCH "${APPEND}" _MATCH "${${VAR}}")
        if(NOT "${_MATCH}" STREQUAL "")
            SET(${VAR} "${${VAR}} ${APPEND}")
        endif()
    endif()
ENDMACRO()


#-----------------------------------------------------------------------
# macro cache_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided and cache it
#
MACRO(CACHE_IFNOT _var _value _type _doc)
  if(NOT ${_var} OR NOT ${CACHE_VARIABLES} MATCHES ${_var})
    set(${_var} ${_value} CACHE ${_type} "${_doc}")
  endif()
ENDMACRO()


#-----------------------------------------------------------------------
# function add_enabled_interface(<NAME>)
#          Mark an interface library as enabled
#
FUNCTION(ADD_ENABLED_INTERFACE _var)
    set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_ENABLED_INTERFACES ${_var})
ENDFUNCTION()


#-----------------------------------------------------------------------
# function add_disabled_interface(<NAME>)
#          Mark an interface as disabled
#
FUNCTION(ADD_DISABLED_INTERFACE _var)
    get_property(_DISABLED GLOBAL PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES)
    if(NOT ${_var} IN_LIST _DISABLED)
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES ${_var})
    endif()
ENDFUNCTION()


#------------------------------------------------------------------------------#
# macro for creating a library target
#
FUNCTION(CREATE_EXECUTABLE)
    # for include dirs, compile flags, definitions, etc. --> use INTERFACE libs
    # and add them to "LINK_LIBRARIES"
    # list of arguments taking multiple values
    set(multival_args
        HEADERS SOURCES PROPERTIES LINK_LIBRARIES INSTALL_DESTINATION)

    # parse args
    cmake_parse_arguments(EXE
        "INSTALL"                    # options
        "TARGET_NAME;"               # single value args
        "${multival_args}"           # multiple value args
        ${ARGN})

    # create library
    add_executable(${EXE_TARGET_NAME} ${EXE_SOURCES} ${EXE_HEADERS})

    # link library
    target_link_libraries(${EXE_TARGET_NAME} ${EXE_LINK_LIBRARIES})

    # target properties
    if(NOT "${EXE_PROPERTIES}" STREQUAL "")
        set_target_properties(${EXE_TARGET_NAME} PROPERTIES ${EXE_PROPERTIES})
    endif()

    if(EXE_INSTALL AND NOT EXE_INSTALL_DESTINATION)
        set(EXE_INSTALL_DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()

    # Install the exe
    if(EXE_INSTALL_DESTINATION)
        install(TARGETS ${EXE_TARGET_NAME} DESTINATION ${EXE_INSTALL_DESTINATION})
    endif()
ENDFUNCTION()

#------------------------------------------------------------------------------#
# macro add_googletest()
#
# Adds a unit test and links against googletest. Additional arguments are linked
# against the test.
#
FUNCTION(ADD_TIMEMORY_GOOGLE_TEST TEST_NAME)
    if(NOT TIMEMORY_BUILD_GOOGLE_TEST AND NOT TIMEMORY_BUILD_TESTING)
        return()
    endif()
    include(GoogleTest)
    # list of arguments taking multiple values
    set(multival_args SOURCES PROPERTIES LINK_LIBRARIES COMMAND OPTIONS ENVIRONMENT)
    # parse args
    cmake_parse_arguments(TEST "DISCOVER_TESTS;ADD_TESTS" "" "${multival_args}" ${ARGN})

    if(NOT TARGET google-test-debug-options)
        add_library(google-test-debug-options INTERFACE)
        target_compile_definitions(google-test-debug-options INTERFACE
            $<$<CONFIG:Debug>:DEBUG> TIMEMORY_TESTING)
    endif()
    list(APPEND TEST_LINK_LIBRARIES google-test-debug-options)

    CREATE_EXECUTABLE(
        TARGET_NAME     ${TEST_NAME}
        OUTPUT_NAME     ${TEST_NAME}
        SOURCES         ${TEST_SOURCES}
        LINK_LIBRARIES  timemory-google-test ${TEST_LINK_LIBRARIES}
        PROPERTIES      "${TEST_PROPERTIES}")

    if("${TEST_COMMAND}" STREQUAL "")
        set(TEST_COMMAND $<TARGET_FILE:${TEST_NAME}>)
    endif()

    if(TEST_DISCOVER_TESTS)
        GTEST_DISCOVER_TESTS(${TEST_NAME}
            ${TEST_OPTIONS})
    elseif(TEST_ADD_TESTS)
        GTEST_ADD_TESTS(TARGET ${TEST_NAME}
            ${TEST_OPTIONS})
    else()
        ADD_TEST(
            NAME                ${TEST_NAME}
            COMMAND             ${TEST_COMMAND}
            WORKING_DIRECTORY   ${CMAKE_CURRENT_LIST_DIR}
            ${TEST_OPTIONS})
        SET_TESTS_PROPERTIES(${TEST_NAME} PROPERTIES ENVIRONMENT "${TEST_ENVIRONMENT}")
    endif()

ENDFUNCTION()

#----------------------------------------------------------------------------------------#
# macro CHECKOUT_GIT_SUBMODULE()
#
#   Run "git submodule update" if a file in a submodule does not exist
#
#   ARGS:
#       RECURSIVE (option) -- add "--recursive" flag
#       RELATIVE_PATH (one value) -- typically the relative path to submodule
#                                    from PROJECT_SOURCE_DIR
#       WORKING_DIRECTORY (one value) -- (default: PROJECT_SOURCE_DIR)
#       TEST_FILE (one value) -- file to check for (default: CMakeLists.txt)
#       ADDITIONAL_CMDS (many value) -- any addition commands to pass
#
FUNCTION(CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE;REPO_URL;REPO_BRANCH"
        "ADDITIONAL_CMDS"
        ${ARGN})

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif()

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif()

    # default assumption
    if(NOT CHECKOUT_REPO_BRANCH)
        set(CHECKOUT_REPO_BRANCH "master")
    endif()

    find_package(Git)
    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        if(NOT CHECKOUT_REPO_URL)
            message(FATAL_ERROR "submodule directory does not exist")
        endif()
    endif()

    # if this file exists --> project has been checked out
    # if not exists --> not been checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")
    # assuming a .gitmodules file exists
    set(_SUBMODULE "${PROJECT_SOURCE_DIR}/.gitmodules")

    set(_TEST_FILE_EXISTS OFF)
    if(EXISTS "${_TEST_FILE}" AND NOT IS_DIRECTORY "${_TEST_FILE}")
        set(_TEST_FILE_EXISTS ON)
    endif()

    if(_TEST_FILE_EXISTS)
        return()
    endif()

    find_package(Git REQUIRED)

    set(_SUBMODULE_EXISTS OFF)
    if(EXISTS "${_SUBMODULE}" AND NOT IS_DIRECTORY "${_SUBMODULE}")
        set(_SUBMODULE_EXISTS ON)
    endif()

    set(_HAS_REPO_URL OFF)
    if(NOT "${CHECKOUT_REPO_URL}" STREQUAL "")
        set(_HAS_REPO_URL ON)
    endif()

    # if the module has not been checked out
    if(NOT _TEST_FILE_EXISTS AND _SUBMODULE_EXISTS)
        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                    ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # check the return code
        if(RET GREATER 0)
            set(_CMD "${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_RELATIVE_PATH}")
            message(STATUS "function(CHECKOUT_GIT_SUBMODULE) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT _TEST_FILE_EXISTS AND _HAS_REPO_URL)
        message(STATUS "Checking out '${CHECKOUT_REPO_URL}' @ '${CHECKOUT_REPO_BRANCH}'...")

        # remove the existing directory
        if(EXISTS "${_DIR}")
            execute_process(COMMAND ${CMAKE_COMMAND} -E remove_directory ${_DIR})
        endif()

        # perform the checkout
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                    ${CHECKOUT_ADDITIONAL_CMDS}
                    ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}
            WORKING_DIRECTORY
                ${CHECKOUT_WORKING_DIRECTORY}
            RESULT_VARIABLE RET)

        # perform the submodule update
        if(CHECKOUT_RECURSIVE AND EXISTS "${_DIR}" AND IS_DIRECTORY "${_DIR}")
            execute_process(
                COMMAND
                    ${GIT_EXECUTABLE} submodule update --init ${_RECURSE}
                WORKING_DIRECTORY
                    ${_DIR}
                RESULT_VARIABLE RET)
        endif()

        # check the return code
        if(RET GREATER 0)
            set(_CMD "${GIT_EXECUTABLE} clone -b ${CHECKOUT_REPO_BRANCH}
                ${CHECKOUT_ADDITIONAL_CMDS} ${CHECKOUT_REPO_URL} ${CHECKOUT_RELATIVE_PATH}")
            message(STATUS "function(CHECKOUT_GIT_SUBMODULE) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        else()
            set(_TEST_FILE_EXISTS ON)
        endif()
    endif()

    if(NOT EXISTS "${_TEST_FILE}" OR NOT _TEST_FILE_EXISTS)
        message(FATAL_ERROR "Error checking out submodule: '${CHECKOUT_RELATIVE_PATH}' to '${_DIR}'")
    endif()

ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# macro to add an interface lib
#
MACRO(ADD_INTERFACE_LIBRARY _TARGET)
    add_library(${_TARGET} INTERFACE ${ARGN})
    cache_list(APPEND ${PROJECT_NAME_UC}_INTERFACE_LIBRARIES ${_TARGET})
    add_enabled_interface(${_TARGET})
ENDMACRO()


#----------------------------------------------------------------------------------------#
# macro to build a library of type: shared, static, object
#
macro(BUILD_LIBRARY)

    # options
    set(_options    PIC NO_CACHE_LIST)
    # single-value
    set(_onevalue   TYPE
                    OUTPUT_NAME
                    TARGET_NAME
                    OUTPUT_DIR
                    LANGUAGE
                    LINKER_LANGUAGE)
    # multi-value
    set(_multival   SOURCES
                    LINK_LIBRARIES
                    COMPILE_DEFINITIONS
                    INCLUDE_DIRECTORIES
                    C_COMPILE_OPTIONS
                    CXX_COMPILE_OPTIONS
                    CUDA_COMPILE_OPTIONS
                    LINK_OPTIONS
                    EXTRA_PROPERTIES)

    cmake_parse_arguments(
        LIBRARY "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    if("${LIBRARY_LANGUAGE}" STREQUAL "")
        set(LIBRARY_LANGUAGE CXX)
    endif()

    if("${LIBRARY_LINKER_LANGUAGE}" STREQUAL "")
        set(LIBRARY_LINKER_LANGUAGE CXX)
    endif()

    if("${LIBRARY_OUTPUT_DIR}" STREQUAL "")
        set(LIBRARY_OUTPUT_DIR ${PROJECT_BINARY_DIR})
    endif()

    if(NOT WIN32 AND NOT XCODE)
        list(APPEND LIBRARY_EXTRA_PROPERTIES
            VERSION                     ${PROJECT_VERSION}
            SOVERSION                   ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
    endif()

    if(NOT WIN32)
        set(LIB_PREFIX )
        list(APPEND LIBRARY_EXTRA_PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR}
            ARCHIVE_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR}
            RUNTIME_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR})
    else()
        set(LIB_PREFIX lib)
    endif()

    # add the library or sources
    if(NOT TARGET ${LIBRARY_TARGET_NAME})
        add_library(${LIBRARY_TARGET_NAME} ${LIBRARY_TYPE} ${LIBRARY_SOURCES})
    else()
        target_sources(${LIBRARY_TARGET_NAME} PRIVATE ${LIBRARY_SOURCES})
    endif()

    # append include directories
    target_include_directories(${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_INCLUDE_DIRECTORIES})

    # compile definitions
    target_compile_definitions(${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_COMPILE_DEFINITIONS})

    # compile flags
    target_compile_options(${LIBRARY_TARGET_NAME}
        PRIVATE
            $<$<COMPILE_LANGUAGE:C>:${LIBRARY_C_COMPILE_OPTIONS}>
            $<$<COMPILE_LANGUAGE:CXX>:${LIBRARY_CXX_COMPILE_OPTIONS}>)

    # cuda flags
    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(CMAKE_CUDA_COMPILER AND "CUDA" IN_LIST LANGUAGES)
        target_compile_options(${LIBRARY_TARGET_NAME}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:${LIBRARY_CUDA_COMPILE_OPTIONS}>)
    endif()

    # link options
    if(NOT CMAKE_VERSION VERSION_LESS 3.13)
        target_link_options(${LIBRARY_TARGET_NAME} PUBLIC ${LIBRARY_LINK_OPTIONS})
    elseif(NOT "${LIBRARY_LINK_OPTIONS}" STREQUAL "")
        list(APPEND LIBRARY_EXTRA_PROPERTIES LINK_OPTIONS ${LIBRARY_LINK_OPTIONS})
    endif()

    # link libraries
    target_link_libraries(${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_LINK_LIBRARIES})

    # other properties
    set_target_properties(
        ${LIBRARY_TARGET_NAME}      PROPERTIES
        OUTPUT_NAME                 ${LIB_PREFIX}${LIBRARY_OUTPUT_NAME}
        LANGUAGE                    ${LIBRARY_LANGUAGE}
        LINKER_LANGUAGE             ${LIBRARY_LINKER_LANGUAGE}
        POSITION_INDEPENDENT_CODE   ${LIBRARY_PIC}
        ${LIBRARY_EXTRA_PROPERTIES})

    if(NOT LIBRARY_NO_CACHE_LIST)
        # add to cached list of compiled libraries
        set(COMPILED_TYPES "SHARED" "STATIC" "MODULE")
        if("${LIBRARY_TYPE}" IN_LIST COMPILED_TYPES)
            cache_list(APPEND ${PROJECT_NAME_UC}_COMPILED_LIBRARIES ${LIBRARY_TARGET_NAME})
        endif()
    endif()
    unset(COMPILED_TYPES)

endmacro()


#----------------------------------------------------------------------------------------#
# finds dependencies
#
function(TIMEMORY_GET_INTERNAL_DEPENDS VAR LINK)
    # set the depends before creating the library so it does not
    # link to itself
    set(DEPENDS)
    foreach(DEP ${ARGN})
        if(TARGET ${DEP})
            list(APPEND DEPENDS ${DEP})
        endif()
        if(TARGET ${DEP}-${LINK})
            list(APPEND DEPENDS ${DEP}-${LINK})
        endif()
        if(TARGET ${DEP}-component-${LINK})
            list(APPEND DEPENDS ${DEP}-component-${LINK})
        endif()
        if(TARGET timemory-${DEP}-${LINK})
            list(APPEND DEPENDS timemory-${DEP}-${LINK})
        endif()
        if(TARGET timemory-${DEP}-component-${LINK})
            list(APPEND DEPENDS timemory-${DEP}-component-${LINK})
        endif()
    endforeach()
    set(${VAR} "${DEPENDS}" PARENT_SCOPE)
endfunction()

#----------------------------------------------------------------------------------------#
# finds dependencies
#
function(TIMEMORY_GET_PROPERTY_DEPENDS VAR LINK)
    # set the depends before creating the library so it does not
    # link to itself
    set(DEPENDS)
    foreach(DEP ${ARGN})
        get_property(TMP GLOBAL PROPERTY TIMEMORY_${LINK}_${DEP}_LIBRARIES)
        list(APPEND DEPENDS ${TMP})
    endforeach()
    set(${VAR} "${DEPENDS}" PARENT_SCOPE)
endfunction()


#----------------------------------------------------------------------------------------#
# require variable
#
function(CHECK_REQUIRED VAR)
    if(NOT DEFINED ${VAR} OR "${${VAR}}" STREQUAL "")
        message(FATAL_ERROR "Variable '${VAR}' must be defined and not empty")
    endif()
endfunction()


#-----------------------------------------------------------------------
# C/C++ development headers
#
macro(INSTALL_HEADER_FILES)
    foreach(_header ${ARGN})
        file(RELATIVE_PATH _relative ${PROJECT_SOURCE_DIR}/source ${_header})
        get_filename_component(_destpath ${_relative} DIRECTORY)
        install(FILES ${_header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${_destpath})
    endforeach()
endmacro()


#----------------------------------------------------------------------------------------#
# macro to build a library of type: shared, static, object
#
macro(BUILD_INTERMEDIATE_LIBRARY)

    # options
    set(_options    USE_INTERFACE
                    INSTALL_SOURCE
                    FORCE_SHARED
                    FORCE_STATIC)
    # single-value
    set(_onevalue   NAME
                    TARGET
                    CATEGORY
                    FOLDER)
    # multi-value
    set(_multival   HEADERS
                    SOURCES
                    DEPENDS
                    PROPERTY_DEPENDS
                    PUBLIC_LINK
                    PRIVATE_LINK)

    cmake_parse_arguments(
        COMP "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    if(WIN32 AND NOT "${COMP_CATEGORY}" STREQUAL "GLOBAL")
        set_property(GLOBAL APPEND PROPERTY TIMEMORY_CXX_LIBRARY_SOURCES
           ${COMP_HEADERS})
        return()
    endif()

    check_required(COMP_NAME)
    check_required(COMP_TARGET)
    check_required(COMP_CATEGORY)
    check_required(COMP_FOLDER)
    check_required(COMP_SOURCES)

    string(TOUPPER "${COMP_NAME}" UPP_COMP)
    string(REPLACE "-" "_" UPP_COMP "${UPP_COMP}")

    set(_LIB_TYPES)
    if(_BUILD_SHARED_CXX OR COMP_FORCE_SHARED)
        list(APPEND _LIB_TYPES shared)
        set(shared_OPTIONS PIC TYPE SHARED)
    endif()

    if(_BUILD_STATIC_CXX OR COMP_FORCE_STATIC)
        list(APPEND _LIB_TYPES static)
        set(static_OPTIONS TYPE STATIC)
    endif()

    set(_SOURCES ${COMP_SOURCES} ${COMP_HEADERS})

    foreach(LINK ${_LIB_TYPES})

        string(TOUPPER "${LINK}" UPP_LINK)
        set(TARGET_NAME timemory-${COMP_TARGET}-${LINK})

        # set the depends before creating the library so it does not
        # link to itself
        timemory_get_internal_depends(DEPENDS ${LINK} ${COMP_DEPENDS})
        timemory_get_property_depends(PROPERTY_DEPENDS ${UPP_LINK} ${COMP_PROPERTY_DEPENDS})

        set_property(GLOBAL APPEND PROPERTY TIMEMORY_HEADERS ${COMP_HEADERS})
        set_property(GLOBAL APPEND PROPERTY TIMEMORY_SOURCES ${COMP_SOURCES})
        set_property(GLOBAL APPEND PROPERTY TIMEMORY_${UPP_LINK}_${COMP_CATEGORY}_LIBRARIES
            timemory-${COMP_TARGET}-${LINK})

        build_library(
            NO_CACHE_LIST
            ${${LINK}_OPTIONS}
            TARGET_NAME         ${TARGET_NAME}
            OUTPUT_NAME         timemory-${COMP_TARGET}
            LANGUAGE            CXX
            LINKER_LANGUAGE     ${_LINKER_LANGUAGE}
            OUTPUT_DIR          ${PROJECT_BINARY_DIR}/${COMP_FOLDER}
            SOURCES             ${_SOURCES}
            CXX_COMPILE_OPTIONS ${${PROJECT_NAME}_CXX_COMPILE_OPTIONS})

        target_link_libraries(${TARGET_NAME} PUBLIC
            timemory-external-${LINK}
            timemory-vector
            ${DEPENDS}
            ${PROPERTY_DEPENDS}
            ${COMP_PUBLIC_LINK})

        target_link_libraries(${TARGET_NAME} PRIVATE
            timemory-compile-options
            timemory-develop-options
            timemory-default-visibility
            timemory-headers
            ${_ANALYSIS_TOOLS}
            ${_ARCH_LIBRARY}
            ${COMP_PRIVATE_LINK})

        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            target_compile_definitions(${TARGET_NAME} PRIVATE DEBUG)
        endif()

        target_compile_definitions(${TARGET_NAME} PRIVATE
            TIMEMORY_SOURCE
            TIMEMORY_${COMP_CATEGORY}_SOURCE
            TIMEMORY_${UPP_COMP}_SOURCE)

        set(_USE_VIS PUBLIC)
        if(COMP_USE_INTERFACE)
            set(_USE_VIS INTERFACE)
        endif()

        target_compile_definitions(${TARGET_NAME} ${_USE_VIS}
            TIMEMORY_USE_${COMP_CATEGORY}_EXTERN
            TIMEMORY_USE_${UPP_COMP}_EXTERN)

        if(WIN32 AND "${LINK}" STREQUAL "shared")
            target_compile_definitions(${TARGET_NAME}
                PRIVATE TIMEMORY_DLL_EXPORT 
                # INTERFACE TIMEMORY_DLL_IMPORT
            )
        endif()

        string(TOLOWER "${COMP_CATEGORY}" LC_CATEGORY)

        install(TARGETS ${TARGET_NAME}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
            EXPORT      ${PROJECT_NAME}-library-depends)

        set_property(GLOBAL APPEND PROPERTY TIMEMORY_INTERMEDIATE_TARGETS ${TARGET_NAME})
        set_property(GLOBAL APPEND PROPERTY TIMEMORY_INTERMEDIATE_${UPP_LINK}_TARGETS
            ${TARGET_NAME})

    endforeach()

    if(COMP_INSTALL_SOURCE)
        install_header_files(${COMP_SOURCES})
    endif()
    
    if(WIN32 AND TARGET timemory-${COMP_TARGET}-shared AND TARGET timemory-${COMP_TARGET}-static)
        add_dependencies(timemory-${COMP_TARGET}-shared timemory-${COMP_TARGET}-static)
    endif()

endmacro()


#-----------------------------------------------------------------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a project feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
FUNCTION(ADD_FEATURE _var _description)
  set(EXTRA_DESC "")
  foreach(currentArg ${ARGN})
      if(NOT "${currentArg}" STREQUAL "${_var}" AND
         NOT "${currentArg}" STREQUAL "${_description}")
          set(EXTRA_DESC "${EXTA_DESC}${currentArg}")
      endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_FEATURES ${_var})
  set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
FUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF("NO_FEATURE" IN_LIST ARGN)
        MARK_AS_ADVANCED(${_NAME})
    ELSE()
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ENDIF()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
FUNCTION(print_enabled_features)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(${_feature})
            # add feature to text
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
            # get description
            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)
            # print description, if not standard ON/OFF, print what is set to
            if(_desc)
                if(NOT "${${_feature}}" STREQUAL "ON" AND
                   NOT "${${_feature}}" STREQUAL "TRUE")
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc} -- [\"${${_feature}}\"]")
                else()
                    string(REGEX REPLACE "^${PROJECT_NAME}_USE_" "" _feature_tmp "${_feature}")
                    string(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    capitalize("${_feature_tmp}" _feature_tmp_c)
                    foreach(_var _feature _feature_tmp _feature_tmp_l _feature_tmp_c)
                        set(_ver "${${${_var}}_VERSION}")
                        if(NOT "${_ver}" STREQUAL "")
                            set(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        endif()
                        unset(_ver)
                    endforeach()
                    set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
                endif()
                set(_desc NOTFOUND)
            endif()
        endif()
    endforeach()

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_disabled_features()
#          Print disabled features plus their docstrings.
#
FUNCTION(print_disabled_features)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY ${PROJECT_NAME}_FEATURES)
    if(NOT "${_features}" STREQUAL "")
        list(REMOVE_DUPLICATES _features)
        list(SORT _features)
    endif()
    foreach(_feature ${_features})
        if(NOT ${_feature})
            set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")

            get_property(_desc GLOBAL PROPERTY ${_feature}_DESCRIPTION)

            if(_desc)
              set(_currentFeatureText "${_currentFeatureText}: ${_desc}")
              set(_desc NOTFOUND)
            endif(_desc)
        endif()
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_enabled_interfaces()
#          Print enabled INTERFACE libraries plus their docstrings.
#
FUNCTION(print_enabled_interfaces)
    set(_basemsg "The following INTERFACE libraries are enabled:")
    set(_currentFeatureText "${_basemsg}")
    get_property(_enabled GLOBAL PROPERTY ${PROJECT_NAME}_ENABLED_INTERFACES)
    get_property(_disabled GLOBAL PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES)
    if(NOT "${_enabled}" STREQUAL "")
        list(REMOVE_DUPLICATES _enabled)
        list(SORT _enabled)
    endif()
    foreach(_LIB ${_disabled})
        if("${_LIB}" IN_LIST _enabled)
            list(REMOVE_ITEM _enabled ${_LIB})
        endif()
    endforeach()
    foreach(_feature ${_enabled})
        # add feature to text
        set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
    endforeach()

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_disabled_interfaces()
#          Print disabled interfaces plus their docstrings.
#
FUNCTION(print_disabled_interfaces)
    set(_basemsg "The following INTERFACE libraries are NOT enabled (empty INTERFACE libraries):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_disabled GLOBAL PROPERTY ${PROJECT_NAME}_DISABLED_INTERFACES)
    if(NOT "${_disabled}" STREQUAL "")
        list(REMOVE_DUPLICATES _disabled)
        list(SORT _disabled)
    endif()
    foreach(_feature ${_disabled})
        set(_currentFeatureText "${_currentFeatureText}\n     ${_feature}")
    endforeach(_feature)

    if(NOT "${_currentFeatureText}" STREQUAL "${_basemsg}")
        message(STATUS "${_currentFeatureText}\n")
    endif()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
# function print_features()
#          Print all features plus their docstrings.
#
FUNCTION(print_features)
    message(STATUS "")
    print_enabled_features()
    print_disabled_features()
    message(STATUS "")
    print_enabled_interfaces()
    print_disabled_interfaces()
ENDFUNCTION()


#----------------------------------------------------------------------------------------#
MACRO(DETERMINE_LIBDIR_DEFAULT VAR)
    set(_LIBDIR_DEFAULT "lib")
    # Override this default 'lib' with 'lib64' iff:
    #  - we are on Linux system but NOT cross-compiling
    #  - we are NOT on debian
    #  - we are on a 64 bits system
    # reason is: amd64 ABI: https://github.com/hjl-tools/x86-psABI/wiki/X86-psABI
    # For Debian with multiarch, use 'lib/${CMAKE_LIBRARY_ARCHITECTURE}' if
    # CMAKE_LIBRARY_ARCHITECTURE is set (which contains e.g. "i386-linux-gnu"
    # and CMAKE_INSTALL_PREFIX is "/usr"
    # See http://wiki.debian.org/Multiarch
    if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
        set(__LAST_LIBDIR_DEFAULT "lib")
        # __LAST_LIBDIR_DEFAULT is the default value that we compute from
        # _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX, not a cache entry for
        # the value that was last used as the default.
        # This value is used to figure out whether the user changed the
        # LIBDIR_DEFAULT value manually, or if the value was the
        # default one. When CMAKE_INSTALL_PREFIX changes, the value is
        # updated to the new default, unless the user explicitly changed it.
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|kFreeBSD|GNU)$"
            AND NOT CMAKE_CROSSCOMPILING)
        if (EXISTS "/etc/debian_version") # is this a debian system ?
            if(CMAKE_LIBRARY_ARCHITECTURE)
                if("${CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$")
                    set(_LIBDIR_DEFAULT "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
                endif()
                if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX
                        AND "${_GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX}" MATCHES "^/usr/?$")
                    set(__LAST_LIBDIR_DEFAULT "lib/${CMAKE_LIBRARY_ARCHITECTURE}")
                endif()
            endif()
        else() # not debian, rely on CMAKE_SIZEOF_VOID_P:
            if(NOT DEFINED CMAKE_SIZEOF_VOID_P)
                message(AUTHOR_WARNING
                    "Unable to determine default LIBDIR_DEFAULT directory "
                    "because no target architecture is known. "
                    "Please enable at least one language before including GNUInstallDirs.")
            else()
                if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
                    set(_LIBDIR_DEFAULT "lib64")
                    if(DEFINED _GNUInstallDirs_LAST_CMAKE_INSTALL_PREFIX)
                        set(__LAST_LIBDIR_DEFAULT "lib64")
                    endif()
                endif()
            endif()
        endif()
    endif()

    # assign the variable
    set(${VAR} "${_LIBDIR_DEFAULT}")
ENDMACRO()

#----------------------------------------------------------------------------------------#
# always determine the default lib directory
DETERMINE_LIBDIR_DEFAULT(LIBDIR_DEFAULT)

cmake_policy(POP)
