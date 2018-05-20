
################################################################################
#
#        timem tool
#
################################################################################

# copy over the python file
configure_file(${CMAKE_CURRENT_LIST_DIR}/timem.py
    ${PROJECT_BINARY_DIR}/pytimem @ONLY)

# timem installation
install(FILES ${PROJECT_BINARY_DIR}/pytimem
    DESTINATION ${TIMEMORY_INSTALL_BINDIR}
    COMPONENT python
    PERMISSIONS
    OWNER_EXECUTE OWNER_READ OWNER_WRITE
    GROUP_EXECUTE GROUP_READ
    WORLD_EXECUTE WORLD_READ)
