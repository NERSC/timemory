# include guard
include_guard(DIRECTORY)

##########################################################################################
#
#        Handles the build settings
#
##########################################################################################


include(GNUInstallDirs)
include(Compilers)


#----------------------------------------------------------------------------------------#
# set the compiler flags
add_c_flag_if_avail("-W")
if(NOT WIN32)
    add_c_flag_if_avail("-Wall")
else()
    add_c_flag_if_avail("/bigobj")
endif()
# add_c_flag_if_avail("-Wextra")
# add_c_flag_if_avail("-Wshadow")
# add_c_flag_if_avail("-Wno-unused-value")
# add_c_flag_if_avail("-Wno-unused-function")
add_c_flag_if_avail("-Wno-unknown-pragmas")
add_c_flag_if_avail("-Wno-ignored-attributes")
# add_c_flag_if_avail("-Wno-reserved-id-macro")
# add_c_flag_if_avail("-Wno-deprecated-declarations")

add_cxx_flag_if_avail("-W")
if(NOT WIN32)
    add_cxx_flag_if_avail("-Wall")
else()
    add_cxx_flag_if_avail("/bigobj")
endif()
# add_cxx_flag_if_avail("-Wextra")
# add_cxx_flag_if_avail("-Wshadow")
# add_cxx_flag_if_avail("-Wno-unused-value")
# add_cxx_flag_if_avail("-Wno-unused-function")
add_cxx_flag_if_avail("-Wno-unknown-pragmas")
add_cxx_flag_if_avail("-Wno-c++17-extensions")
add_cxx_flag_if_avail("-Wno-ignored-attributes")
# add_cxx_flag_if_avail("-Wno-implicit-fallthrough")
# add_cxx_flag_if_avail("-Wno-deprecated-declarations")
add_cxx_flag_if_avail("-Wno-attributes")

if(NOT CMAKE_CXX_COMPILER_IS_GNU)
    # these flags succeed with GNU compiler but are unknown (clang flags)
    # add_cxx_flag_if_avail("-Wno-exceptions")
    # add_cxx_flag_if_avail("-Wno-class-memaccess")
    # add_cxx_flag_if_avail("-Wno-reserved-id-macro")
    # add_cxx_flag_if_avail("-Wno-unused-private-field")
else()
    add_cxx_flag_if_avail("-Wno-class-memaccess")
    add_cxx_flag_if_avail("-Wno-cast-function-type")
endif()

#----------------------------------------------------------------------------------------#
# non-debug optimizations
#
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND TIMEMORY_BUILD_EXTRA_OPTIMIZATIONS)
    add_flag_if_avail("-finline-functions")
    add_flag_if_avail("-funroll-loops")
    add_flag_if_avail("-ftree-vectorize")
    add_flag_if_avail("-ftree-loop-optimize")
    add_flag_if_avail("-ftree-loop-vectorize")
    # add_flag_if_avail("-freciprocal-math")
    # add_flag_if_avail("-fno-signed-zeros")
    # add_flag_if_avail("-mfast-fp")
else()
    add_cxx_flag_if_avail("-ftemplate-backtrace-limit=0")
endif()

#----------------------------------------------------------------------------------------#
# debug-safe optimizations
#
add_cxx_flag_if_avail("-faligned-new")
if(NOT TIMEMORY_USE_SANITIZER)
    add_cxx_flag_if_avail("-ftls-model=${TIMEMORY_TLS_MODEL}")
endif()

if(TIMEMORY_BUILD_LTO)
    add_c_flag_if_avail("-flto")
    add_cxx_flag_if_avail("-flto")
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
    add_target_flag_if_avail(timemory-develop-options "-Wshadow")
endif()

#----------------------------------------------------------------------------------------#
# architecture optimizations
#
add_interface_library(timemory-vector)
add_interface_library(timemory-arch)
add_interface_library(timemory-avx512)
target_link_libraries(timemory-avx512 INTERFACE timemory-arch)
if(TIMEMORY_USE_ARCH)
    target_link_libraries(timemory-compile-options INTERFACE timemory-arch)
endif()
# always provide vectorization width
target_link_libraries(timemory-compile-options INTERFACE timemory-vector)

find_package(CpuArch)

if(CpuArch_FOUND)

    set(_VEC_256 OFF)
    set(_VEC_512 OFF)
    foreach(_ARCH ${CpuArch_FEATURES})
        if("${_ARCH}" MATCHES ".*avx512.*" OR "${_ARCH}" MATCHES ".*AVX512.*")
            set(_VEC_512 ON)
        elseif("${_ARCH}" MATCHES ".*avx.*" OR "${_ARCH}" MATCHES ".*AVX.*")
            set(_VEC_256 ON)
        endif()
    endforeach()

    if(_VEC_512)
        message(STATUS "Compiling with vector width: 512")
        target_compile_definitions(timemory-vector INTERFACE TIMEMORY_VEC=512)
    elseif(_VEC_256)
        message(STATUS "Compiling with vector width: 256")
        target_compile_definitions(timemory-vector INTERFACE TIMEMORY_VEC=256)
    else()
        message(STATUS "Compiling with vector width: 128")
        target_compile_definitions(timemory-vector INTERFACE TIMEMORY_VEC=128)
    endif()

    foreach(_ARCH ${CpuArch_FEATURES})
        # intel compiler
        if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
            add_target_flag_if_avail(timemory-arch "-x${_ARCH}")
        endif()
        # non-intel compilers
        if(NOT CMAKE_C_COMPILER_IS_INTEL OR NOT CMAKE_CXX_COMPILER_IS_INTEL)
            add_target_flag_if_avail(timemory-arch "-m${_ARCH}")
        endif()
    endforeach()

endif()

if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
    add_target_flag_if_avail(timemory-avx512 "-xMIC-AVX512")
endif()

if(NOT CMAKE_C_COMPILER_IS_INTEL OR NOT CMAKE_CXX_COMPILER_IS_INTEL)
    add_target_flag_if_avail(timemory-avx512 "-mavx512f" "-mavx512pf" "-mavx512er" "-mavx512cd")
endif()


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
# -------------------------------------------------------------------------------------- #
# user customization
#
add_user_flags(timemory-compile-options "C")
add_user_flags(timemory-compile-options "CXX")
