
add_subdirectory(ex1)

foreach(_FILE test.py nested.py)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${_FILE}
        ${PROJECT_BINARY_DIR}/${_FILE} COPYONLY)
endforeach(_FILE test.py)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_LIST_DIR} ${CMAKE_BINARY_DIR}/examples)

set_property(GLOBAL APPEND PROPERTY
    BUILDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR})
