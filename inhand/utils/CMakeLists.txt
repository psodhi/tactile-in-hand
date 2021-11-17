# Install headers
file(GLOB utils_headers "*.h")
install(FILES ${utils_headers} DESTINATION include/inhand/utils)

# Build tests
gtsamAddTestsGlob(utils "tests/*.cpp" "" ${PROJECT_NAME})