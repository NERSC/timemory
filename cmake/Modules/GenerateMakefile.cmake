

set_property(DIRECTORY PROPERTY PROCESSED )

#----------------------------------------------------------------------------------------#

function(TIMEMORY_MAKEFILE_TARGET _VAR _TARG)
    string(REPLACE "-" "_" _FILE_TARG "${_TARG}")
    string(REGEX REPLACE ".*::" "" _FILE_TARG "${_FILE_TARG}")
    set(${_VAR} ${_FILE_TARG} PARENT_SCOPE)
endfunction()

#----------------------------------------------------------------------------------------#

function(TIMEMORY_MAKEFILE_TARGET_NAME _VAR _TARG)
    string(REPLACE "-" "_" _FILE_TARG "${_TARG}")
    string(REGEX REPLACE ".*::" "" _FILE_TARG "${_FILE_TARG}")
    string(TOUPPER "${_FILE_TARG}" _NAME)
    set(${_VAR} ${_NAME} PARENT_SCOPE)
endfunction()

#----------------------------------------------------------------------------------------#

function(TIMEMORY_MAKEFILE_TARGET_EXTRACT _TARG _TYPE)

    get_property(PROCESSED DIRECTORY PROPERTY PROCESSED)
    if("${_TARG}" IN_LIST PROCESSED)
        return()
    endif()
    set_property(DIRECTORY APPEND PROPERTY PROCESSED ${_TARG})

    timemory_makefile_target(_FILE_TARG ${_TARG})
    timemory_makefile_target_name(_NAME ${_TARG})

    set(C_OPTS)
    set(CXX_OPTS)
    set(CUDA_OPTS)
    set(EXTRA_OPTS)
    set(INCLUDE)
    set(DEFINE)
    set(LINKED)
    set(DEPEND)

    get_target_property(INCLUDE_DIRS ${_TARG} ${_TYPE}_INCLUDE_DIRECTORIES)
    get_target_property(COMPILE_OPTS ${_TARG} ${_TYPE}_COMPILE_OPTIONS)
    get_target_property(COMPILE_DEFS ${_TARG} ${_TYPE}_COMPILE_DEFINITIONS)
    get_target_property(LIBRARY_LINK ${_TARG} ${_TYPE}_LINK_LIBRARIES)
    get_target_property(LIBRARY_LDIR ${_TARG} ${_TYPE}_LINK_DIRECTORIES)

    if(INCLUDE_DIRS)
        foreach(_DIR ${INCLUDE_DIRS})
            if(_DIR MATCHES ".*<BUILD_INTERFACE:")
                continue()
            endif()
            if(_DIR MATCHES ".*<INSTALL_INTERFACE:")
                string(REPLACE "\$" "" _DIR "${_DIR}")
                string(REPLACE "<INSTALL_INTERFACE:" "" _DIR "${_DIR}")
                string(REPLACE ">" "" _DIR "${_DIR}")
                if(NOT EXISTS "${_DIR}")
                    set(_DIR "${CMAKE_INSTALL_PREFIX}/${_DIR}")
                endif()
            endif()
            list(APPEND INCLUDE "-I${_DIR}")
        endforeach()

        string(REPLACE ";" " " INCLUDE "${INCLUDE}")
        # message(STATUS "[${_FILE_TARG}]> Include: ${INCLUDE}")
    endif()

    if(COMPILE_OPTS)

        foreach(_OPT ${COMPILE_OPTS})
            if(_OPT MATCHES ".*COMPILE_LANGUAGE:C>.*")

                string(REPLACE "\$" "" _OPT "${_OPT}")
                string(REPLACE "<<COMPILE_LANGUAGE:C>:" "" _OPT "${_OPT}")
                string(REPLACE ">" "" _OPT "${_OPT}")
                list(APPEND C_OPTS ${_OPT})

            elseif(_OPT MATCHES ".*COMPILE_LANGUAGE:CXX>:*")

                string(REPLACE "\$" "" _OPT "${_OPT}")
                string(REPLACE "<<COMPILE_LANGUAGE:CXX>:" "" _OPT "${_OPT}")
                string(REPLACE ">" "" _OPT "${_OPT}")
                list(APPEND CXX_OPTS ${_OPT})

            elseif(_OPT MATCHES ".*COMPILE_LANGUAGE:CUDA>:*")

                string(REPLACE "\$" "" _OPT "${_OPT}")
                string(REPLACE "<<COMPILE_LANGUAGE:CUDA>:" "" _OPT "${_OPT}")
                string(REPLACE ">" "" _OPT "${_OPT}")
                list(APPEND CUDA_OPTS ${_OPT})

            else()

                list(APPEND EXTRA_OPTS ${_OPT})

            endif()

        endforeach()

        foreach(_TYPE C CXX CUDA EXTRA)
            string(REPLACE ";" " " ${_TYPE}_OPTS "${${_TYPE}_OPTS}")
        endforeach()

        # message(STATUS "[${_FILE_TARG}]> Compile C     : ${C_OPTS}")
        # message(STATUS "[${_FILE_TARG}]> Compile CXX   : ${CXX_OPTS}")
        # message(STATUS "[${_FILE_TARG}]> Compile CUDA  : ${CUDA_OPTS}")
        # message(STATUS "[${_FILE_TARG}]> Compile EXTRA : ${EXTRA_OPTS}")
    endif()

    if(COMPILE_DEFS)
        foreach(_DEF ${COMPILE_DEFS})
            list(APPEND DEFINE "-D${_DEF}")
        endforeach()
        string(REPLACE ";" " " DEFINE "${DEFINE}")
        # message(STATUS "[${_FILE_TARG}]> Defines: ${DEFINE}")
    endif()

    if(LIBRARY_LINK)
        foreach(_LINK ${LIBRARY_LDIR})
            if(EXISTS ${_LINK})
                list(APPEND LINKED "-L${_LDIR}")
            endif()
        endforeach()
    endif()

    if(LIBRARY_LINK)
        foreach(_LINK ${LIBRARY_LINK})
            if(TARGET ${_LINK})
                timemory_makefile_target_extract(${_LINK} INTERFACE)
                timemory_makefile_target_name(_DEP ${_LINK})
                list(APPEND DEPEND ${_DEP})
            else()
                if(EXISTS "${_LINK}")
                    get_filename_component(_LDIR "${_LINK}" PATH)
                    get_filename_component(_LNAME "${_LINK}" NAME_WE)
                    string(REGEX REPLACE "^lib" "" _LNAME "${_LNAME}")
                    list(APPEND LINKED "-L${_LDIR}")
                    list(APPEND LINKED "-l${_LNAME}")
                else()
                    list(APPEND LINKED "-l${_LINK}")
                endif()
            endif()
        endforeach()
        string(REPLACE ";" " " LINKED "${LINKED}")
        # message(STATUS "[${_FILE_TARG}]> Library: ${LINKED}")
    endif()

    file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "#
#   TARGET: ${_NAME}
#

")

    if(INCLUDE)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_INCLUDE = ${INCLUDE}\n")
    endif()

    if(LINKED)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_LIBS = ${LINKED}\n")
    endif()

    if(DEFINE)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_DEFS = ${DEFINE}\n")
    endif()

    if(C_OPTS)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_CFLAGS = ${C_OPTS}\n")
    endif()

    if(CXX_OPTS)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_CPPFLAGS = ${CXX_OPTS}\n")
    endif()

    if(CUDA_OPTS)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_CUFLAGS = ${CUDA_OPTS}\n")
    endif()

    if(DEPEND)
        string(REPLACE ";" " " DEPEND "${DEPEND}")
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "${_NAME}_DEPENDS = ${DEPEND}\n")
    endif()

    if(INCLUDE OR LINKED OR DEFINE OR C_OPTS OR CXX_OPTS OR CUDA_OPTS OR DEPEND)
        file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "\n")
    endif()

endfunction()

#----------------------------------------------------------------------------------------#

message(STATUS "")

if(TIMEMORY_MAKEFILE_COMPILED_LIBRARIES)
    list(REMOVE_DUPLICATES TIMEMORY_MAKEFILE_COMPILED_LIBRARIES)
endif()

if(TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES)
    list(REMOVE_DUPLICATES TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES)
endif()

set(MAKEFILE_LIBS)
foreach(_LIB ${TIMEMORY_MAKEFILE_COMPILED_LIBRARIES} ${TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES})
    timemory_makefile_target_name(_LIB ${_LIB})
    list(APPEND MAKEFILE_LIBS ${_LIB})
endforeach()
string(REPLACE ";" " " MAKEFILE_LIBS "${MAKEFILE_LIBS}")

file(WRITE ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc
"
#
# This file was auto-generated by the timemory CMake build system
#

TIMEMORY_MAKEFILE_LIBRARIES = ${MAKEFILE_LIBS}

")

#----------------------------------------------------------------------------------------#

foreach(_TARG ${TIMEMORY_MAKEFILE_COMPILED_LIBRARIES})
    timemory_makefile_target_extract(${_TARG} INTERFACE)
endforeach()

foreach(_TARG ${TIMEMORY_MAKEFILE_INTERFACE_LIBRARIES})
    timemory_makefile_target_extract(${_TARG} INTERFACE)
endforeach()

file(APPEND ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc "\n")

#----------------------------------------------------------------------------------------#

message(STATUS "")

#----------------------------------------------------------------------------------------#

install(FILES ${PROJECT_BINARY_DIR}/output/Makefile.timemory.inc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
