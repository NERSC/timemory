
if(WIN32)
    set_directory_properties(PROPERTIES
        COMPILE_DEFINITIONS _TIMEMORY_DLL)
endif()

set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})

set(_GLOBAL)
set(_EXTERN)
set(_ENABLED)
set(_DISABLE)

#
# handles dependencies and option
#
macro(TIMEMORY_HANDLE_GENERIC FOLDER _TARG)
    list(APPEND ${_TARG} ${FOLDER})
    set(SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${FOLDER})
    file(GLOB_RECURSE ${FOLDER}_header_files
        ${SOURCE_PATH}/*.hpp)

    if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
        file(GLOB_RECURSE ${FOLDER}_source_files
            ${SOURCE_PATH}/*.cpp
            ${SOURCE_PATH}/*.cu)
    else()
        file(GLOB_RECURSE ${FOLDER}_source_files
            ${SOURCE_PATH}/*.cpp)
    endif()

    set(${FOLDER}_DEPENDS ${ARGN} PARENT_SCOPE)
    set(${FOLDER}_HEADERS "${${FOLDER}_header_files}" PARENT_SCOPE)
    set(${FOLDER}_SOURCES "${${FOLDER}_source_files}" PARENT_SCOPE)
endmacro()

#
# handles dependencies and option
#
macro(TIMEMORY_HANDLE_CATEGORY OPT FOLDER)
    if(DEFINED ${OPT} AND "${${OPT}}")

        list(APPEND _ENABLED ${FOLDER})
        set(SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/components/${FOLDER})
        file(GLOB_RECURSE ${FOLDER}_header_files
            ${SOURCE_PATH}/*.hpp)

        if(TIMEMORY_USE_CUDA AND CMAKE_CUDA_COMPILER)
            file(GLOB_RECURSE ${FOLDER}_source_files
                ${SOURCE_PATH}/*.cpp
                ${SOURCE_PATH}/*.cu)
        else()
            file(GLOB_RECURSE ${FOLDER}_source_files
                ${SOURCE_PATH}/*.cpp)
        endif()

        set(${FOLDER}_DEPENDS ${ARGN} PARENT_SCOPE)
        set(${FOLDER}_HEADERS "${${FOLDER}_header_files}" PARENT_SCOPE)
        set(${FOLDER}_SOURCES "${${FOLDER}_source_files}" PARENT_SCOPE)

    else()
        list(APPEND _DISABLED ${FOLDER})
    endif()
endmacro()

#
#   Shorthand options
#
set(TIMEMORY_USE_ALWAYS ON)
set(TIMEMORY_USE_GPERFTOOLS ${TIMEMORY_USE_GPERF})
if(TIMEMORY_USE_PAPI OR TIMEMORY_USE_CUPTI)
    set(TIMEMORY_USE_ROOFLINE ON)
endif()

#
#   global and extern
#
timemory_handle_generic(hash         _GLOBAL)
timemory_handle_generic(environment  _GLOBAL  timemory-hash)
timemory_handle_generic(settings     _GLOBAL  timemory-hash timemory-environment)
timemory_handle_generic(manager      _GLOBAL  timemory-hash timemory-environment timemory-settings)

set(_EXTERN_DEPENDS timemory-hash timemory-environment timemory-settings timemory-manager)

timemory_handle_generic(containers   _EXTERN  ${_EXTERN_DEPENDS})
timemory_handle_generic(storage      _EXTERN  ${_EXTERN_DEPENDS})
timemory_handle_generic(plotting     _EXTERN  ${_EXTERN_DEPENDS})

#
#   native
#
timemory_handle_category(TIMEMORY_USE_ALWAYS  rusage)
timemory_handle_category(TIMEMORY_USE_ALWAYS  timing)
timemory_handle_category(TIMEMORY_USE_ALWAYS  trip_count)

#
#   have extern deps
#
# timemory_handle_category(TIMEMORY_USE_ALLINEA allinea)
timemory_handle_category(TIMEMORY_USE_CALIPER       caliper     timemory-caliper)
timemory_handle_category(TIMEMORY_USE_CUDA          cuda        timemory-cuda)
timemory_handle_category(TIMEMORY_USE_CUPTI         cupti       timemory-cupti)
timemory_handle_category(TIMEMORY_USE_GOTCHA        gotcha      timemory-gotcha)
timemory_handle_category(TIMEMORY_USE_GPERFTOOLS    gperftools  timemory-gperftools)
timemory_handle_category(TIMEMORY_USE_LIKWID        likwid      timemory-likwid)
timemory_handle_category(TIMEMORY_USE_PAPI          papi        timemory-papi)
timemory_handle_category(TIMEMORY_USE_ROOFLINE      roofline    timemory-papi timemory-cupti
    timemory-cpu-roofline timemory-gpu-roofline timemory-roofline)
timemory_handle_category(TIMEMORY_USE_TAU           tau_marker  timemory-tau)
timemory_handle_category(TIMEMORY_USE_VTUNE         vtune       timemory-vtune)

#
#   full dependence
#
timemory_handle_category(TIMEMORY_USE_ALWAYS  user_bundle ${_ENABLED})


set(TIMEMORY_GLOBAL "${_GLOBAL}" PARENT_SCOPE)
set(TIMEMORY_EXTERN "${_EXTERN}" PARENT_SCOPE)
set(TIMEMORY_ENABLED_COMPONENTS "${_ENABLED}" PARENT_SCOPE)
set(TIMEMORY_DISABLED_COMPONENTS "${_DISABLED}" PARENT_SCOPE)
