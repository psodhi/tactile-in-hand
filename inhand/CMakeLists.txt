set(inhand_subdirs 
dynamics
utils
)
set(inhand_srcs)

# files want to be excluded
set(excluded_sources "")

# Library sources
foreach(subdir ${inhand_subdirs})
  file(GLOB subdir_srcs "${subdir}/*.cpp" "${subdir}/*.h")
  list(REMOVE_ITEM subdir_srcs "${excluded_sources}")

  file(GLOB subdir_test_files "${subdir}/tests/*")
  
  list(APPEND inhand_srcs ${subdir_srcs})
  message(STATUS "Building Module: ${subdir}")

  # local and tests
  add_subdirectory(${subdir})
endforeach(subdir)

# shared
message(STATUS "Build shared library")
add_library(${PROJECT_NAME} SHARED ${inhand_srcs})
target_link_libraries(inhand ${Boost_LIBRARIES} ${GTSAM_LIBRARIES})
set_target_properties(inhand PROPERTIES
  OUTPUT_NAME         ${PROJECT_NAME}
  LINKER_LANGUAGE CXX
  CLEAN_DIRECT_OUTPUT 1)
  install(TARGETS ${PROJECT_NAME} EXPORT inhand-exports LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)

list(APPEND INHAND_EXPORTED_TARGETS inhand)
set(INHAND_EXPORTED_TARGETS "${INHAND_EXPORTED_TARGETS}" PARENT_SCOPE)