
#include <cstdlib>
#include <iostream>

void printUsage(char const* progName) {
  std::cerr << "Usage: " << progName << " <input-file> <output-file> <iso>\n";
  std::cerr << "  input-file   Input file, supported formats: *.rek\n";
  std::cerr << "  output-file  Output mesh file, *.ply format\n";
  std::cerr << "  iso          ISO threshold for surface extraction.\n";
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

