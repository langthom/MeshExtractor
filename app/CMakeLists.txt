
add_executable(pme_cli
  ExtractionPipeline.h
  Main.cpp
)

# The pipeline runs its stages on separate threads, and the CPU extraction backend spreads a batch
# of chunks over the cores with OpenMP.
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(pme_cli PRIVATE pme pme_io pme_mesh_io Threads::Threads OpenMP::OpenMP_CXX)
