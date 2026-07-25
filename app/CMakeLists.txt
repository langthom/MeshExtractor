

add_executable(pme_cli
  Main.cpp
)

target_link_libraries(pme_cli PRIVATE pme pme_io)

