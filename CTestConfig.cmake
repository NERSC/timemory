#=============================================================================
# CMake - Cross Platform Makefile Generator
# Copyright 2000-2009 Kitware, Inc., Insight Software Consortium
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

# if CTEST_SITE was not provided
if(NOT DEFINED CTEST_SITE)
    # [A] if CTEST_SITE set at configure
    set(_HOSTNAME "@CTEST_SITE@")
    # [B] if CTEST_SITE not set at configure, grab the HOSTNAME
    if("${_HOSTNAME}" STREQUAL "")
        execute_process(COMMAND uname -n
            OUTPUT_VARIABLE _HOSTNAME
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif("${_HOSTNAME}" STREQUAL "")
    # either [A] or [B]
    set(CTEST_SITE              "${_HOSTNAME}")
    unset(_HOSTNAME)
endif(NOT DEFINED CTEST_SITE)

set(CTEST_PROJECT_NAME          "TiMemory")
set(CTEST_NIGHTLY_START_TIME    "01:00:00 PDT")
set(CTEST_DROP_METHOD           "http")
set(CTEST_DROP_SITE             "jonathan-madsen.info")
set(CTEST_DROP_LOCATION         "/cdash/public/submit.php?project=TiMemory")
set(CTEST_DROP_SITE_CDASH       TRUE)
set(CTEST_CDASH_VERSION         "1.6")
set(CTEST_CDASH_QUERY_VERSION   TRUE)
