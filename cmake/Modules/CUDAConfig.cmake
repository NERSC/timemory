#
#  Configures CUDA for timemory
#

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

if("CUDA" IN_LIST LANGUAGES)

    find_package(CUDA REQUIRED)

    target_compile_definitions(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
        ${PROJECT_USE_CUDA_OPTION})
    target_include_directories(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    set_target_properties(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda PROPERTIES
        INTERFACE_CUDA_STANDARD                 ${CMAKE_CUDA_STANDARD}
        INTERFACE_CUDA_STANDARD_REQUIRED        ${CMAKE_CUDA_STANDARD_REQUIRED}
        INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS   ON
        INTERFACE_CUDA_SEPARABLE_COMPILATION    ON)

    set(CUDA_AUTO_ARCH "auto")
    set(CUDA_ARCHITECTURES auto kepler tesla maxwell pascal volta turing)
    set(TIMEMORY_CUDA_ARCH "${CUDA_AUTO_ARCH}" CACHE STRING
        "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
    add_feature(TIMEMORY_CUDA_ARCH "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
    set_property(CACHE TIMEMORY_CUDA_ARCH PROPERTY STRINGS ${CUDA_ARCHITECTURES})

    set(cuda_kepler_arch    30)
    set(cuda_tesla_arch     35)
    set(cuda_maxwell_arch   50)
    set(cuda_pascal_arch    60)
    set(cuda_volta_arch     70)
    set(cuda_turing_arch    75)

    if(NOT "${TIMEMORY_CUDA_ARCH}" STREQUAL "${CUDA_AUTO_ARCH}")
        if(NOT "${TIMEMORY_CUDA_ARCH}" IN_LIST CUDA_ARCHITECTURES)
            message(WARNING
                "CUDA architecture \"${TIMEMORY_CUDA_ARCH}\" not known. Options: ${TIMEMORY_CUDA_ARCH}")
            unset(TIMEMORY_CUDA_ARCH CACHE)
            set(TIMEMORY_CUDA_ARCH "${CUDA_AUTO_ARCH}")
        else()
            set(_ARCH_NUM ${cuda_${TIMEMORY_CUDA_ARCH}_arch})
            if(DEFINED PROJECT_CUDA_USE_HALF_OPTION AND _ARCH_NUM LESS 60)
                set(${PROJECT_CUDA_USE_HALF_OPTION} OFF)
            endif()
        endif()
    endif()

    if(NOT _ARCH_NUM)
        set(_ARCH_NUM 60)
    endif()

    target_compile_options(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${_ARCH_NUM}
        -gencode=arch=compute_${_ARCH_NUM},code=sm_${_ARCH_NUM}
        -gencode=arch=compute_${_ARCH_NUM},code=compute_${_ARCH_NUM}>)

    #   30, 32      + Kepler support
    #               + Unified memory programming
    #   35          + Dynamic parallelism support
    #   50, 52, 53  + Maxwell support
    #   60, 61, 62  + Pascal support
    #   70, 72      + Volta support
    #   75          + Turing support

    # target_compile_options(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
    #    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

    add_user_flags(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda "CUDA")

    target_compile_options(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

    if(DEFINED PROJECT_CUDA_USE_HALF_OPTION)
        if(${PROJECT_CUDA_USE_HALF_OPTION} AND NOT _ARCH_NUM LESS 60)
            target_compile_definitions(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
                ${PROJECT_CUDA_USE_HALF_DEFINITION})
        endif()
    endif()

    if(NOT WIN32)
        get_filename_component(_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        target_compile_options(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda-compiler INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${_COMPILER_DIR}>)
    endif()

    target_include_directories(${PROJECT_CUDA_INTERFACE_PREFIX}-cuda INTERFACE
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    find_library(CUDA_dl_LIBRARY
        NAMES dl)

    #target_compile_options(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart INTERFACE
    #    $<$<COMPILE_LANGUAGE:CUDA>:--cudart=shared>)

    #target_compile_options(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-static INTERFACE
    #    $<$<COMPILE_LANGUAGE:CUDA>:--cudart=static>)

    target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart INTERFACE
        ${CUDA_CUDART_LIBRARY} ${CUDA_rt_LIBRARY})

    target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-device INTERFACE
        ${CUDA_cudadevrt_LIBRARY} ${CUDA_rt_LIBRARY})

    target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-static INTERFACE
        ${CUDA_cudart_static_LIBRARY} ${CUDA_rt_LIBRARY})

    if(CUDA_dl_LIBRARY)
        target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart INTERFACE
            ${CUDA_dl_LIBRARY})

        target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-device INTERFACE
            ${CUDA_dl_LIBRARY})

        target_link_libraries(${PROJECT_CUDA_INTERFACE_PREFIX}-cudart-static INTERFACE
            ${CUDA_dl_LIBRARY})
    endif()

else()
    message(FATAL_ERROR
        "CUDA is not supported! ${PROJECT_USE_CUDA_OPTION}=${${PROJECT_USE_CUDA_OPTION}}")
endif()
