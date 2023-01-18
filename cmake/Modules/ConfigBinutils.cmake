# ======================================================================================
# binutils.cmake
#
# Configure binutils for Dyninst
#
# ----------------------------------------
#
# Directly exports the following CMake variables
#
# binutils_ROOT_DIR       - Computed base directory the of binutils installation
# binutils_LIBRARY_DIRS   - Link directories for binutils libraries binutils_LIBRARIES
# - binutils library files binutils_INCLUDE - binutils include files
#
# NOTE: The exported binutils_ROOT_DIR can be different from the value provided by the
# user in the case that it is determined to build binutils from source. In such a case,
# binutils_ROOT_DIR will contain the directory of the from-source installation.
#
# See Modules/Findbinutils.cmake for details
#
# ======================================================================================

include_guard(GLOBAL)

set(TPL_STAGING_PREFIX ${PROJECT_BINARY_DIR}/external/tpls)
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${TPL_STAGING_PREFIX}/lib)

# always provide Dyninst::binutils even if it is empty
add_library(binutils::binutils IMPORTED INTERFACE)

if(NOT UNIX)
    return()
endif()

timemory_message(STATUS "Attempting to build binutils as external project")

set(_TIMEMORY_BINUTILS_BUILD_BYPRODUCTS
    ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external
    ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/include
    ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/bfd
    ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/opcodes
    ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/libsframe
    ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/libiberty
    ${PROJECT_BINARY_DIR}/external/tpls/lib/libbfd.a
    ${PROJECT_BINARY_DIR}/external/tpls/lib/libopcodes.a
    ${PROJECT_BINARY_DIR}/external/tpls/lib/libiberty.a
    ${PROJECT_BINARY_DIR}/external/tpls/lib/libsframe.a)

foreach(_NAME bfd opcodes iberty)
    list(APPEND _TIMEMORY_BINUTILS_BUILD_BYPRODUCTS ${_FILE})
endforeach()

find_program(
    MAKE_COMMAND
    NAMES make gmake
    PATH_SUFFIXES bin REQUIRED)

include(ExternalProject)
externalproject_add(
    binutils-external
    PREFIX ${PROJECT_BINARY_DIR}/external/binutils
    URL http://ftp.gnu.org/gnu/binutils/binutils-2.40.tar.gz
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CFLAGS=-fPIC\ -O3
        CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=-fPIC\ -O3 <SOURCE_DIR>/configure
        --prefix=${TPL_STAGING_PREFIX}
    BUILD_COMMAND ${MAKE_COMMAND} all-libiberty all-bfd all-opcodes all-libelf
                  all-libsframe
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${_TIMEMORY_BINUTILS_BUILD_BYPRODUCTS}")

add_custom_command(
    TARGET binutils-external
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${TPL_STAGING_PREFIX}/lib
    COMMAND
        install ARGS -C
        ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/bfd/libbfd.a
        ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/opcodes/libopcodes.a
        ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/libiberty/libiberty.a
        ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/libsframe/.libs/libsframe.a
        ${TPL_STAGING_PREFIX}/lib/
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external
    COMMENT "Installing binutils...")

foreach(_NAME bfd opcodes iberty sframe)
    set(_FILE
        ${TPL_STAGING_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        )

    add_library(binutils::${_NAME}-library STATIC IMPORTED)
    set_property(TARGET binutils::${_NAME}-library PROPERTY IMPORTED_LOCATION ${_FILE})
    add_dependencies(binutils::${_NAME}-library binutils-external)
endforeach()

find_package(ZLIB)

target_include_directories(
    binutils::binutils
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/bfd>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/opcodes>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/binutils/src/binutils-external/libiberty>
    )
target_link_libraries(
    binutils::binutils
    INTERFACE $<BUILD_INTERFACE:binutils::bfd-library>
              $<BUILD_INTERFACE:binutils::iberty-library>
              $<BUILD_INTERFACE:binutils::opcodes-library>
              $<BUILD_INTERFACE:binutils::sframe-library>
              $<BUILD_INTERFACE:$<IF:$<TARGET_EXISTS:ZLIB::ZLIB>,ZLIB::ZLIB,z>>
              $<BUILD_INTERFACE:${CMAKE_DL_LIBS}>)

unset(_TIMEMORY_BINUTILS_BUILD_BYPRODUCTS)
