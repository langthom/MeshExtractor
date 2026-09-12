
# The GPU tests call into the CUDA runtime directly, e.g. to observe device memory, so the test
# executable needs the runtime itself rather than only the library that was built against it.
find_package(CUDAToolkit REQUIRED)

# Collect all unit tests in a single executable.
add_executable(pme_tests
  TestsMain.cpp
  MHDIOTest.cpp
  VolumeMetaDataTest.cpp
  ChunkifierTest.cpp
  SlabScheduleTest.cpp
  MeshAssemblerTest.cpp
  PLYStreamWriterTest.cpp
  MarchingCubesTablesTest.cpp
  MeshAssertions.h
  ReferenceMarchingCubes.h
  ChunkMeshExtractorTest.cpp
  ChunkMeshAssemblyTest.cpp
  PLYReader.h
  PipelineEndToEndTest.cpp
)

# Link the library this depends on.
target_link_libraries(pme_tests PRIVATE pme pme_io pme_mesh_io CUDA::cudart)

# Register with CTest to run it this way.
include(CTest)
add_test(NAME AllUnitTests COMMAND pme_tests)

# The extraction cases need a CUDA device. They already skip themselves when none is present, but
# they also sit in their own doctest suite, so a machine without a card can leave them out
# altogether rather than running them only to have each one bail.
add_test(NAME HostUnitTests COMMAND pme_tests --test-suite-exclude=gpu)
set_tests_properties(AllUnitTests PROPERTIES LABELS "gpu")
