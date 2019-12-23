# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Handles the build settings
#
##########################################################################################


include(GNUInstallDirs)
include(Compilers)


find_library(dl_LIBRARY NAMES dl)
if(dl_LIBRARY)
    target_link_libraries(timemory-compile-options INTERFACE ${dl_LIBRARY})
endif()

if(WIN32)
    set(OS_FLAG "/bigobj")
else()
    set(OS_FLAG "-Wall")
endif()

#----------------------------------------------------------------------------------------#
# set the compiler flags
add_flag_if_avail(
    "-W" "${OS_FLAG}" "-Wno-unknown-pragmas" "-Wno-ignored-attributes"
    "-Wno-attributes" "-Wno-cast-function-type"
    "-Wno-unused-command-line-argument")

add_cxx_flag_if_avail("-Wno-class-memaccess")

if(TIMEMORY_BUILD_QUIET)
    add_flag_if_avail("-Wno-unused-value" "-Wno-unused-function"
        "-Wno-unknown-pragmas" "-Wno-reserved-id-macro" "-Wno-deprecated-declarations"
        "-Wno-implicit-fallthrough" "-Wno-deprecated-declarations")
endif()

if(NOT CMAKE_CXX_COMPILER_IS_GNU)
    # these flags succeed with GNU compiler but are unknown (clang flags)
    # add_cxx_flag_if_avail("-Wno-exceptions")
    # add_cxx_flag_if_avail("-Wno-reserved-id-macro")
    # add_cxx_flag_if_avail("-Wno-unused-private-field")
else()
    # add_cxx_flag_if_avail("-Wno-class-memaccess")
endif()

#----------------------------------------------------------------------------------------#
# non-debug optimizations
#
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS)
    add_flag_if_avail("-finline-functions" "-funroll-loops"
        "-ftree-vectorize" "-ftree-loop-optimize" "-ftree-loop-vectorize")
    # add_flag_if_avail("-freciprocal-math" "-fno-signed-zeros" "-mfast-fp")
endif()

#----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")
if(NOT TIMEMORY_USE_SANITIZER)
    add_cxx_flag_if_avail("-ftls-model=${TIMEMORY_TLS_MODEL}")
endif()

if(TIMEMORY_BUILD_LTO)
    add_flag_if_avail("-flto")
endif()

#----------------------------------------------------------------------------------------#
# print compilation timing reports (Clang compiler)
#
add_interface_library(timemory-compile-timing)
add_target_flag_if_avail(timemory-compile-timing "-ftime-report")
if(TIMEMORY_USE_COMPILE_TIMING)
    target_link_libraries(timemory-compile-options INTERFACE timemory-compile-timing)
endif()

#----------------------------------------------------------------------------------------#
# developer build flags
#
add_interface_library(timemory-develop-options)
if(TIMEMORY_BUILD_DEVELOPER)
    add_target_flag_if_avail(timemory-develop-options "-Wshadow" "-Wextra")
endif()

#----------------------------------------------------------------------------------------#
# developer build flags
#
if(dl_LIBRARY)
    # This instructs the linker to add all symbols, not only used ones, to the dynamic
    # symbol table. This option is needed for some uses of dlopen or to allow obtaining
    # backtraces from within a program.
    add_flag_if_avail("-rdynamic")
endif()

#----------------------------------------------------------------------------------------#
# architecture optimizations
#
add_interface_library(timemory-vector)
add_interface_library(timemory-arch)
target_link_libraries(timemory-compile-options INTERFACE timemory-vector)

set(VECTOR_DEFINITION               TIMEMORY_VEC)
set(VECTOR_INTERFACE_TARGET         timemory-vector)
set(ARCH_INTERFACE_TARGET           timemory-arch)

include(ArchConfig)

#----------------------------------------------------------------------------------------#
# sanitizer
#
if(TIMEMORY_USE_SANITIZER)
    set(SANITIZER_TYPES address memory thread leak)

    set(asan_key "address")
    set(msan_key "memory")
    set(tsan_key "thread")
    set(lsan_key "leak")

    set(address_lib asan)
    set(memory_lib msan)
    set(thread_lib tsan)
    set(leak_lib lsan)

    find_library(SANITIZER_asan_LIBRARY NAMES asan)
    find_library(SANITIZER_msan_LIBRARY NAMES msan)
    find_library(SANITIZER_tsan_LIBRARY NAMES tsan)
    find_library(SANITIZER_lsan_LIBRARY NAMES lsan)

    string(TOLOWER "${SANITIZER_TYPE}" SANITIZER_TYPE)
    list(REMOVE_ITEM SANITIZER_TYPES ${SANITIZER_TYPE})
    set(SANITIZER_TYPES ${SANITIZER_TYPE} ${SANITIZER_TYPES})

    foreach(_TYPE ${SANITIZER_TYPES})
        set(_LIB ${${_TYPE}_lib})
        add_interface_library(timemory-${_TYPE}-sanitizer)
        add_target_flag_if_avail(timemory-${_TYPE}-sanitizer "-fsanitize=${SANITIZER_TYPE}")
        target_link_libraries(timemory-${_TYPE}-sanitizer INTERFACE ${SANITIZER_${_LIB}_LIBRARY})
    endforeach()

    foreach(_TYPE ${SANITIZER_TYPE} ${SANITIZER_TYPES})
        set(_LIB ${${_TYPE}_lib})
        if((c_timemory_${_TYPE}_sanitizer_fsanitize_${SANITIZER_TYPE} OR
                cxx_timemory_${_TYPE}_sanitizer_fsanitize_${SANITIZER_TYPE}) AND
                SANITIZER_${_LIB}_LIBRARY)
            add_interface_library(timemory-sanitizer)
            add_target_flag_if_avail(timemory-sanitizer "-fno-omit-frame-pointer")
            target_compile_definitions(timemory-sanitizer INTERFACE TIMEMORY_USE_SANITIZER)
            target_link_libraries(timemory-sanitizer INTERFACE timemory-${_TYPE}-sanitizer)
            break()
        else()
            message(STATUS "${_TYPE} sanitizer not found. library: ${SANITIZER_${_LIB}_LIBRARY}...")
        endif()
    endforeach()

    if(NOT TARGET timemory-sanitizer)
        message(WARNING "TIMEMORY_USE_SANITIZER not found. Tried: ${SANITIZER_TYPES}")
        unset(SANITIZER_TYPE CACHE)
        set(TIMEMORY_USE_SANITIZER OFF)
    endif()

endif()


#----------------------------------------------------------------------------------------#
# user customization
#
add_user_flags(timemory-compile-options "C")
add_user_flags(timemory-compile-options "CXX")
add_user_flags(timemory-compile-options "CUDA")
