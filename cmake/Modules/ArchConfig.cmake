#
# Configures architecture options
#

find_package(CpuArch)

if(CpuArch_FOUND)

    if(VECTOR_INTERFACE_TARGET AND VECTOR_DEFINITION)
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
            target_compile_definitions(${VECTOR_INTERFACE_TARGET} INTERFACE ${VECTOR_DEFINITION}=512)
        elseif(_VEC_256)
            message(STATUS "Compiling with vector width: 256")
            target_compile_definitions(${VECTOR_INTERFACE_TARGET} INTERFACE TIMEMORY_VEC=256)
        else()
            message(STATUS "Compiling with vector width: 128")
            target_compile_definitions(${VECTOR_INTERFACE_TARGET} INTERFACE TIMEMORY_VEC=128)
        endif()
    endif()

    if(ARCH_INTERFACE_TARGET)
        foreach(_ARCH ${CpuArch_FEATURES})
            add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "-m${_ARCH}")
        endforeach()

        # intel compiler
        if(CpuArch_FEATURES)
            if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
                list(LENGTH CpuArch_FEATURES NUM_FEATURES)
                math(EXPR LAST_FEATURE_IDX "${NUM_FEATURES}-1")
                list(GET CpuArch_FEATURES ${LAST_FEATURE_IDX} _ARCH)
                if(_ARCH)
                    add_target_flag_if_avail(${ARCH_INTERFACE_TARGET} "-ax${_ARCH}")
                endif()
                unset(_ARCH)
                unset(NUM_FEATURES)
                unset(LAST_FEATURE_IDX)
            endif()
        endif()
    endif()

endif()
