# MacroUtilities - useful macros and functions for generic tasks
#
# CMake Extensions
# ----------------
# macro set_ifnot(<var> <value>)
#       If variable var is not set, set its value to that provided
#
# function enum_option(<option>
#                      VALUES <value1> ... <valueN>
#                      TYPE   <valuetype>
#                      DOC    <docstring>
#                      [DEFAULT <elem>]
#                      [CASE_INSENSITIVE])
#          Declare a cache variable <option> that can only take values
#          listed in VALUES. TYPE may be FILEPATH, PATH or STRING.
#          <docstring> should describe that option, and will appear in
#          the interactive CMake interfaces. If DEFAULT is provided,
#          <elem> will be taken as the zero-indexed element in VALUES
#          to which the value of <option> should default to if not
#          provided. Otherwise, the default is taken as the first
#          entry in VALUES. If CASE_INSENSITIVE is present, then
#          checks of the value of <option> against the allowed values
#          will ignore the case when performing string comparison.
#
#
# General
# --------------
# function add_feature(<NAME> <DOCSTRING>)
#          Add a  feature, whose activation is specified by the
#          existence of the variable <NAME>, to the list of enabled/disabled
#          features, plus a docstring describing the feature
#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
#

# - Include guard
if(__timemory_macroutilities_isloaded)
  return()
endif()
set(__timemory_macroutilities_isloaded YES)

cmake_policy(PUSH)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    cmake_policy(SET CMP0054 NEW)
endif()

include(CMakeDependentOption)
include(CMakeParseArguments)


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
# GENERAL
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

  set_property(GLOBAL APPEND PROPERTY PROJECT_FEATURES ${_var})
  set_property(GLOBAL PROPERTY ${_var}_DESCRIPTION "${_description}${EXTRA_DESC}")
ENDFUNCTION()


#------------------------------------------------------------------------------#
# function add_option(<OPTION_NAME> <DOCSRING> <DEFAULT_SETTING> [NO_FEATURE])
#          Add an option and add as a feature if NO_FEATURE is not provided
#
FUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)
    SET(_FEATURE ${ARGN})
    OPTION(${_NAME} "${_MESSAGE}" ${_DEFAULT})
    IF(NOT "${_FEATURE}" STREQUAL "NO_FEATURE")
        ADD_FEATURE(${_NAME} "${_MESSAGE}")
    ELSE()
        MARK_AS_ADVANCED(${_NAME})
    ENDIF()
ENDFUNCTION(ADD_OPTION _NAME _MESSAGE _DEFAULT)


#------------------------------------------------------------------------------#
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
MACRO(CHECKOUT_GIT_SUBMODULE)
    # parse args
    cmake_parse_arguments(
        CHECKOUT
        "RECURSIVE"
        "RELATIVE_PATH;WORKING_DIRECTORY;TEST_FILE"
        "ADDITIONAL_CMDS"
        ${ARGN})

    if(NOT CHECKOUT_WORKING_DIRECTORY)
        set(CHECKOUT_WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    endif(NOT CHECKOUT_WORKING_DIRECTORY)

    if(NOT CHECKOUT_TEST_FILE)
        set(CHECKOUT_TEST_FILE "CMakeLists.txt")
    endif(NOT CHECKOUT_TEST_FILE)

    set(_DIR "${CHECKOUT_WORKING_DIRECTORY}/${CHECKOUT_RELATIVE_PATH}")
    # ensure the (possibly empty) directory exists
    if(NOT EXISTS "${_DIR}")
        message(FATAL_ERROR "submodule directory does not exist")
    endif(NOT EXISTS "${_DIR}")

    # if this file exists --> project has been checked out
    # if not exists --> not been checked out
    set(_TEST_FILE "${_DIR}/${CHECKOUT_TEST_FILE}")

    # if the module has not been checked out
    if(NOT EXISTS "${_TEST_FILE}")
        find_package(Git REQUIRED)

        set(_RECURSE )
        if(CHECKOUT_RECURSIVE)
            set(_RECURSE --recursive)
        endif(CHECKOUT_RECURSIVE)

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
            message(STATUS "macro(CHECKOUT_SUBMODULE) failed.")
            message(FATAL_ERROR "Command: \"${_CMD}\"")
        endif()

    endif()

ENDMACRO()


#------------------------------------------------------------------------------#
# macro to add an interface lib
#
MACRO(ADD_INTERFACE_LIBRARY _TARGET)
    add_library(${_TARGET} INTERFACE)
    list(APPEND EXTERNAL_LIBRARIES ${_TARGET})
    list(APPEND INSTALL_LIBRARIES ${_TARGET})
    list(APPEND INTERFACE_LIBRARIES ${_TARGET})
ENDMACRO()


#------------------------------------------------------------------------------#
# macro to add an interface lib
#
MACRO(ADD_EXPORTED_INTERFACE_LIBRARY _TARGET)
    add_library(${_TARGET} INTERFACE)
    list(APPEND INSTALL_LIBRARIES ${_TARGET})
    list(APPEND INTERFACE_LIBRARIES ${_TARGET})
ENDMACRO()


#------------------------------------------------------------------------------#
# macro to build a library of type: shared, static, object
#
macro(BUILD_LIBRARY)

    # options
    set(_options    PIC)
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
                    EXTRA_PROPERTIES)

    cmake_parse_arguments(
        LIBRARY "${_options}" "${_onevalue}" "${_multival}" ${ARGN})

    if(NOT WIN32 AND NOT XCODE)
        list(APPEND LIBRARY_EXTRA_PROPERTIES
            VERSION                     ${PROJECT_VERSION}
            SOVERSION                   ${PROJECT_VERSION_MAJOR})
    endif()

    if(NOT WIN32)
        set(LIB_PREFIX )
        list(APPEND LIBRARY_EXTRA_PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY    ${LIBRARY_OUTPUT_DIR})
    else()
        set(LIB_PREFIX lib)
    endif()

    add_library(${LIBRARY_TARGET_NAME}
        ${LIBRARY_TYPE} ${LIBRARY_SOURCES})

    target_include_directories(${LIBRARY_TARGET_NAME}
        PUBLIC ${EXTERNAL_INCLUDE_DIRS}
        PRIVATE ${${PROJECT_NAME}_TARGET_INCLUDE_DIRS})

    target_compile_definitions(${LIBRARY_TARGET_NAME}
        PUBLIC ${LIBRARY_COMPILE_DEFINITIONS})

    target_compile_options(${LIBRARY_TARGET_NAME}
        PRIVATE
            $<$<COMPILE_LANGUAGE:C>:${${PROJECT_NAME}_C_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:${${PROJECT_NAME}_CXX_FLAGS}>)

    if(NOT LIBRARY_TYPE STREQUAL OBJECT)
        target_link_libraries(${LIBRARY_TARGET_NAME}
            PUBLIC ${LIBRARY_LINK_LIBRARIES})
    endif()
    
    set_target_properties(
        ${LIBRARY_TARGET_NAME}          PROPERTIES
        OUTPUT_NAME                     ${LIB_PREFIX}${LIBRARY_OUTPUT_NAME}
        LANGUAGE                        ${LIBRARY_LANGUAGE}
        LINKER_LANGUAGE                 ${LIBRARY_LINKER_LANGUAGE}
        POSITION_INDEPENDENT_CODE       ${LIBRARY_PIC}
        ${LIBRARY_EXTRA_PROPERTIES})

    list(APPEND INSTALL_LIBRARIES ${LIBRARY_TARGET_NAME})

endmacro(BUILD_LIBRARY)


#------------------------------------------------------------------------------#
# function print_enabled_features()
#          Print enabled  features plus their docstrings.
#
FUNCTION(print_enabled_features)
    set(_basemsg "The following features are defined/enabled (+):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
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
                    string(REGEX REPLACE "^USE_" "" _feature_tmp "${_feature}")
                    string(TOLOWER "${_feature_tmp}" _feature_tmp_l)
                    capitalize("${_feature_tmp}" _feature_tmp_c)
                    foreach(_var _feature_tmp _feature_tmp_l _feature_tmp_c)
                        set(_ver "${${${_var}}_VERSION}")
                        if(NOT "${_ver}" STREQUAL "")
                            set(_desc "${_desc} -- [found version ${_ver}]")
                            break()
                        endif()
                        unset(_ver)
                    endforeach(_var _feature_tmp _feature_tmp_l _feature_tmp_c)
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


#------------------------------------------------------------------------------#
# function print_disabled_features()
#          Print disabled features plus their docstrings.
#
FUNCTION(print_disabled_features)
    set(_basemsg "The following features are NOT defined/enabled (-):")
    set(_currentFeatureText "${_basemsg}")
    get_property(_features GLOBAL PROPERTY PROJECT_FEATURES)
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

#------------------------------------------------------------------------------#
# function print_features()
#          Print all features plus their docstrings.
#
FUNCTION(print_features)
    message(STATUS "")
    print_enabled_features()
    print_disabled_features()
ENDFUNCTION()


#------------------------------------------------------------------------------#
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

    # if assign to another variable
    if(NOT "${VAR}" STREQUAL "LIBDIR_DEFAULT")
        set(${VAR} "${_LIBDIR_DEFAULT}")
    endif(NOT "${VAR}" STREQUAL "LIBDIR_DEFAULT")

    # cache the value
    if(NOT DEFINED LIBDIR_DEFAULT)
        set(LIBDIR_DEFAULT "${_LIBDIR_DEFAULT}" CACHE PATH "Object code libraries (${_LIBDIR_DEFAULT})" FORCE)
    elseif(DEFINED __LAST_LIBDIR_DEFAULT
            AND "${__LAST_LIBDIR_DEFAULT}" STREQUAL "${LIBDIR_DEFAULT}")
        set_property(CACHE LIBDIR_DEFAULT PROPERTY VALUE "${_LIBDIR_DEFAULT}")
    endif()
ENDMACRO()

#------------------------------------------------------------------------------#
# always determine the default lib directory
DETERMINE_LIBDIR_DEFAULT(LIBDIR_DEFAULT)

cmake_policy(POP)
