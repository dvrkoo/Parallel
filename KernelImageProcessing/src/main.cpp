#include "BenchmarkSuite.h"
#include <iostream>

int main(int argc, char **argv) {
  try {
    BenchmarkSuite suite(argc, argv);
    suite.run();
  } catch (const std::exception &e) {
    std::cerr << "An unexpected error occurred: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "An unknown error occurred." << std::endl;
    return 1;
  }
  return 0;
}
