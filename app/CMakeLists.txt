

add_executable(pme_cli
  ExtractionPipeline.h
  Main.cpp
)

target_link_libraries(pme_cli PRIVATE pme pme_io pme_mesh_io)
