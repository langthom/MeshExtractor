
# Collect all unit tests in a single executable.
add_executable(pme_tests
  TestsMain.cpp
  MHDIOTest.cpp
  VolumeMetaDataTest.cpp
  ChunkifierTest.cpp
)

# Link the library this depends on.
target_link_libraries(pme_tests PRIVATE pme pme_io)

# Register with CTest to run it this way.
include(CTest)
add_test(NAME AllUnitTests COMMAND pme_tests)

