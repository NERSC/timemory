
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

macro(CHECK_REQUIRED VAR)
    if(NOT DEFINED ${VAR})
        message(FATAL_ERROR "Error! Variable '${VAR}' must be defined")
    endif()
endmacro()

macro(SET_DEFAULT VAR)
    if(NOT DEFINED ${VAR})
        set(${VAR} ${ARGN})
    endif()
endmacro()

CHECK_REQUIRED(FOLDER)

STRING(TOUPPER "${FOLDER}" FOLDER_UPPER)

SET_DEFAULT(INPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/template)
SET_DEFAULT(OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/${FOLDER})

# default values
SET_DEFAULT(CHECK ${FOLDER_UPPER})

if("${INPUT_DIR}" STREQUAL "${OUTPUT_DIR}")
    message(FATAL_ERROR "Input directory == Output directory (${INPUT_DIR} == ${OUTPUT_DIR})")
endif()

if(EXISTS "${OUTPUT_DIR}")
    message(FATAL_ERROR "Output directory (${OUTPUT_DIR}) exists!")
endif()

file(GLOB_RECURSE INPUT_FILES ${INPUT_DIR}/*.in)

foreach(INP ${INPUT_FILES})
    string(REPLACE "${CMAKE_CURRENT_LIST_DIR}/template/" "${OUTPUT_DIR}/" OUT "${INP}")
    string(REPLACE ".in" "" OUT "${OUT}")
    message(STATUS "${INP} --> ${OUT}")
    configure_file(${INP} ${OUT} @ONLY)
endforeach()
