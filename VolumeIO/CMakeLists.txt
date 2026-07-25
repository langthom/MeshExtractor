
add_library(pme_io
  SliceChunkedVolumeDataIO.h
  VolumeMetaData.h
  VolumeMetaData.cpp
  # Specialities for supported formats.
  SliceChunkedMHDIO.h
  SliceChunkedMHDIO.cpp
)


