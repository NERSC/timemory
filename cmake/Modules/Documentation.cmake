#
# Create a "make doc" target using Doxygen
# Prototype:
#     GENERATE_DOCUMENTATION(doxygen_config_file)
# Parameters:
#    doxygen_config_file    Doxygen configuration file (must in the
# root of the source directory)

include(MacroUtilities)
include(CMakeDependentOption)

#------------------------------------------------------------------------------#

if(TIMEMORY_SETUP_PY AND TIMEMORY_DOXYGEN_DOCS)
    message(AUTHOR_WARNING "Skipping documentation generation because TIMEMORY_SETUP_PY=ON")
    set(TIMEMORY_DOXYGEN_DOCS OFF CACHE BOOL "Make a `doc` make target" FORCE)
endif(TIMEMORY_SETUP_PY AND TIMEMORY_DOXYGEN_DOCS)

#------------------------------------------------------------------------------#

if(TIMEMORY_DOXYGEN_DOCS)
    # if BUILD_DOXYGEN_DOCS = ON, we want to build docs quietly
    # else, don't build quietly
    CMAKE_DEPENDENT_OPTION(TIMEMORY_DOXYGEN_DOCS_QUIET
        "Suppress standard output when making the docs" ON
        "BUILD_DOXYGEN_DOCS" OFF)
    mark_as_advanced(TIMEMORY_DOXYGEN_DOCS_QUIET)

    if(TIMEMORY_DOXYGEN_DOCS_QUIET)
        set(DOXYGEN_QUIET YES)
    else()
        set(DOXYGEN_QUIET NO)
    endif()

    # GraphViz dot program is used to build call graphs, caller graphs, class graphs
    find_program(GRAPHVIZ_DOT_PATH dot)
    mark_as_advanced(GRAPHVIZ_DOT_PATH)

    if("${GRAPHVIZ_DOT_PATH}" STREQUAL "GRAPHVIZ_DOT_PATH-NOTFOUND")
        set(DOXYGEN_DOT_FOUND NO)
        set(GRAPHVIZ_DOT_PATH "")
    else()
        set(DOXYGEN_DOT_FOUND YES)
        set(GRAPHVIZ_DOT_PATH ${GRAPHVIZ_DOT_PATH})
    endif()

    # available Doxygen doc formats
    set(AVAILABLE_DOXYGEN_DOC_FORMATS HTML LATEX MAN XML RTF)
    # we want HTML, LATEX, and MAN to be default
    set(_default_on "MAN")
    foreach(_doc_format ${AVAILABLE_DOXYGEN_DOC_FORMATS})
        # find if we want it on
        STRING(REGEX MATCH "${_doc_format}" SET_TO_ON "${_default_on}")
        # if doc format is MAN and it is not a UNIX machine --> turn off
        if("${_doc_format}" STREQUAL "MAN" AND NOT UNIX)
            set(SET_TO_ON "")
        endif()
        # set ON/OFF
        if("${SET_TO_ON}" STREQUAL "")
            set(_default "OFF")
        else()
            set(_default "ON")
        endif()
        # add option
        option(ENABLE_DOXYGEN_${_doc_format}_DOCS
            "Build documentation with ${_doc_format} format" ${_default})
        mark_as_advanced(ENABLE_DOXYGEN_${_doc_format}_DOCS)
    endforeach()

    # loop over doc formats and set GENERATE_DOXYGEN_${_doc_format}_DOC to
    # YES/NO GENERATE_DOXYGEN_${_doc_format}_DOC is used in configure_file @ONLY
    foreach(_doc_format ${AVAILABLE_DOXYGEN_DOC_FORMATS})
        if(ENABLE_DOXYGEN_${_doc_format}_DOCS)
            set(GENERATE_DOXYGEN_${_doc_format}_DOC YES)
        else()
            set(GENERATE_DOXYGEN_${_doc_format}_DOC NO)
        endif()
    endforeach()


    if(DOXYGEN_DOT_FOUND)
        set(DOXYGEN_DOT_GRAPH_TYPES CLASS CALL CALLER)
        # options to turn generation of class, call, and caller graphs
        foreach(_graph_type ${DOXYGEN_DOT_GRAPH_TYPES})
            # create CMake doc string
            string(TOLOWER _graph_type_desc ${_graph_type})
            # add option
            option(ENABLE_DOXYGEN_${_graph_type}_GRAPH "${_message}" ON)
            mark_as_advanced(ENABLE_DOXYGEN_${_graph_type}_GRAPH)
            # set GENERATE_DOXYGEN_${_graph_type}_GRAPH to YES/NO
            # GENERATE_DOXYGEN_${_graph_type}_GRAPH is used in configure_file
            # @ONLY
            if(ENABLE_DOXYGEN_${_graph_type}_GRAPH)
                set(GENERATE_DOXYGEN_${_graph_type}_GRAPH YES)
            else()
                set(GENERATE_DOXYGEN_${_graph_type}_GRAPH NO)
            endif()
        endforeach()
    endif()

    # get the documentation include directories
    get_property(BUILDTREE_INCLUDE_DIRS GLOBAL
                 PROPERTY TIMEMORY_DOCUMENTATION_INCLUDE_DIRS)

    if("${BUILDTREE_INCLUDE_DIRS}" STREQUAL "")
        message(FATAL_ERROR "Property TIMEMORY_DOCUMENTATION_INCLUDE_DIRS is empty")
    endif()

    LIST(REMOVE_DUPLICATES BUILDTREE_INCLUDE_DIRS)
    # if the headers are in include/ and sources are in src/
    set(BUILDTREE_SOURCE_DIRS)
    foreach(_dir ${BUILDTREE_INCLUDE_DIRS})
        get_filename_component(_path "${_dir}" PATH)
        get_filename_component(_name "${_dir}" NAME)
        get_filename_component(_path "${_path}" ABSOLUTE)
        if("${_name}" STREQUAL "include")
            set(_src_dir "${_path}/src")
        endif()
        if(EXISTS "${_src_dir}")
            list(APPEND BUILDTREE_SOURCE_DIRS ${_src_dir})
        endif()
    endforeach()

    # define BUILDTREE_DIRS which is used in configure_file @ONLY
    set(BUILDTREE_DIRS ${BUILDTREE_INCLUDE_DIRS})
    # if there were src/ folders, add them
    if(NOT "${BUILDTREE_SOURCE_DIRS}" STREQUAL "")
        LIST(REMOVE_DUPLICATES BUILDTREE_SOURCE_DIRS)
        LIST(APPEND BUILDTREE_DIRS ${BUILDTREE_SOURCE_DIRS})
    endif()

    LIST(REMOVE_DUPLICATES BUILDTREE_DIRS)
    # Doxyfiles was spaces not semi-colon separated lists
    STRING(REPLACE ";" " " BUILDTREE_DIRS "${BUILDTREE_DIRS}")

    #-----------------------------------------------------------------------

    if(NOT "${SUBPROJECT}" STREQUAL "")
        string(TOUPPER ${SUBPROJECT} SUBPROJECT_U)
        option(INCLUDE_${SUBPROJECT_U}_IN_DOCS "Include ${SUBPROJECT} source files in documentation" OFF)
        mark_as_advanced(INCLUDE_${SUBPROJECT_U}_IN_DOCS)

        if(INCLUDE_${SUBPROJECT_U}_IN_DOCS)
            set(_${SUBPROJECT}_DIRS ${${SUBPROJECT}_ROOT_SOURCE_DIR})
            set(_SUB_DIRS ${_${SUBPROJECT}_DIRS})
            while(NOT "${_SUB_DIRS}" STREQUAL "")
                set(_loop_sub_dirs )
                foreach(_dir ${_SUB_DIRS})
                    list_subdirectories(_sub ${_dir} FALSE)
                    if(NOT "${_sub}" STREQUAL "")
                        list(APPEND _${SUBPROJECT}_DIRS ${_sub})
                        list(APPEND _loop_sub_dirs ${_sub})
                    endif()
                endforeach()
                set(_SUB_DIRS ${_loop_sub_dirs})
            endwhile()

            list(REMOVE_DUPLICATES _${SUBPROJECT}_DIRS)
            STRING(REPLACE ";" " " ${SUBPROJECT_U}_SOURCE_DIRS "${_${SUBPROJECT}_DIRS}")
            set(BUILDTREE_DIRS "${BUILDTREE_DIRS} ${${SUBPROJECT}_SOURCE_DIRS}")

        endif()
    endif()
    #-----------------------------------------------------------------------

    if(XCODE)
        set(GENERATE_DOCSET_IF_XCODE YES)
    else()
        set(GENERATE_DOCSET_IF_XCODE NO)
    endif()

    #-----------------------------------------------------------------------

    configure_file(${PROJECT_SOURCE_DIR}/cmake/Templates/Doxyfile.in
                   ${PROJECT_BINARY_DIR}/doc/Doxyfile.${PROJECT_NAME}
                   @ONLY)

    if(ENABLE_DOXYGEN_HTML_DOCS)
      FILE(WRITE ${PROJECT_BINARY_DIR}/doc/${PROJECT_NAME}_Documentation.html
          "<meta http-equiv=\"refresh\" content=\"1;url=html/index.html\">")
    endif()

endif() # TIMEMORY_DOXYGEN_DOCS

#------------------------------------------------------------------------------#
# Macro to generate documentation
# from:
#   http://www.cmake.org/pipermail/cmake/2007-May/014174.html
MACRO(GENERATE_DOCUMENTATION DOXYGEN_CONFIG_FILE)

    FIND_PACKAGE(Doxygen)
    if(NOT Doxygen_FOUND)
     message(STATUS "Doxygen executable cannot be found. Disable TIMEMORY_DOXYGEN_DOCS")
	 return()
    endif()
    SET(DOXYFILE_FOUND false)

    IF(EXISTS ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE})
        SET(DOXYFILE_FOUND true)
    ELSE(EXISTS ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE})
        MESSAGE(STATUS "Doxygen config file was not found at ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE}")
    ENDIF(EXISTS ${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE})

    IF( DOXYGEN_FOUND )
        IF( DOXYFILE_FOUND )
            # Add target
            if(TIMEMORY_BUILD_DOXYGEN_DOCS)
                ADD_CUSTOM_TARGET(doc ALL ${DOXYGEN_EXECUTABLE}
                                  "${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE}" )
            else()
                ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE}
                                  "${PROJECT_BINARY_DIR}/doc/${DOXYGEN_CONFIG_FILE}" )
            endif()

            install(DIRECTORY   ${PROJECT_BINARY_DIR}/doc/man/
                    DESTINATION ${CMAKE_INSTALL_MANDIR}
                    OPTIONAL
                    COMPONENT documentation
            )

            install(DIRECTORY   ${PROJECT_BINARY_DIR}/doc/html
                    DESTINATION ${CMAKE_INSTALL_DOCDIR}
                    OPTIONAL
                    COMPONENT documentation
            )

            install(DIRECTORY   ${PROJECT_BINARY_DIR}/doc/latex
                    DESTINATION ${CMAKE_INSTALL_DOCDIR}
                    OPTIONAL
                    COMPONENT documentation
            )

            install(FILES       ${PROJECT_BINARY_DIR}/doc/${PROJECT_NAME}_Documentation.html
                    DESTINATION ${CMAKE_INSTALL_DOCDIR}
                    OPTIONAL
                    COMPONENT documentation
            )

        ELSE( DOXYFILE_FOUND )
            MESSAGE( STATUS "Doxygen configuration file not found - Documentation will not be generated" )
        ENDIF( DOXYFILE_FOUND )
    ELSE(DOXYGEN_FOUND)
        MESSAGE(STATUS "Doxygen not found - Documentation will not be generated")
    ENDIF(DOXYGEN_FOUND)

ENDMACRO(GENERATE_DOCUMENTATION)

#------------------------------------------------------------------------------#
# Macro to generate PDF manual from LaTeX using pdflatex
# assumes manual is in ${CMAKE_SOURCE_DIR}/doc
MACRO(GENERATE_MANUAL MANUAL_TEX MANUAL_BUILD_PATH EXTRA_FILES_TO_COPY)

    find_program(PDFLATEX pdflatex)

    if(PDFLATEX AND NOT "${PDFLATEX}" STREQUAL "PDFLATEX-NOTFOUND")
        # name with no path is given
        set(MANUAL_NAME ${MANUAL_TEX})
        # set to full path
        set(MANUAL_BUILD_PATH ${CMAKE_BINARY_DIR}/${MANUAL_BUILD_PATH})

        if(NOT EXISTS ${CMAKE_SOURCE_DIR}/doc/${MANUAL_TEX})
            message(FATAL_ERROR "LaTeX of manual for ${PROJECT_NAME} is not in ${CMAKE_SOURCE_DIR}/doc")
        endif()

        configure_file(${CMAKE_SOURCE_DIR}/doc/${MANUAL_TEX}
                       ${MANUAL_BUILD_PATH}/${MANUAL_NAME}
                       COPYONLY)

        foreach(_file ${EXTRA_FILES_TO_COPY})
            configure_file(${CMAKE_SOURCE_DIR}/doc/${_file}
                           ${MANUAL_BUILD_PATH}/${_file}
                           COPYONLY)
        endforeach()

        add_custom_target(man
                            ${PDFLATEX} "${MANUAL_NAME}"
                          WORKING_DIRECTORY
                            ${MANUAL_BUILD_PATH}
        )
    endif()
ENDMACRO()
