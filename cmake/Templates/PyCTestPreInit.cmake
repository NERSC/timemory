#
# remove the Init.cmake so that original Stages.cmake does not continue
#
execute_process(COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_LIST_DIR}/Init.cmake
                OUTPUT_QUIET)

#
# execute the custom stages
#
ctest_run_script("${CMAKE_CURRENT_LIST_DIR}/CustomStages.cmake" RETURN_VALUE stages_ret)

if(NOT stages_ret EQUAL 0)
    message(FATAL_ERROR "CTest failed with exit code: ${stages_ret}")
endif()
