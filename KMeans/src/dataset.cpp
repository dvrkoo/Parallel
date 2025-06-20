#include "dataset.h"
#include <fstream>
#include <random>
#include <vector>

void generate_dataset_csv(int n_samples, int dim, const std::string &filename) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(0.0, 10.0);

  std::ofstream out(filename);
  // header (optional)
  for (int j = 0; j < dim; ++j) {
    out << "x" << j;
    if (j + 1 < dim)
      out << ",";
  }
  out << "\n";

  // rows
  std::vector<double> point(dim);
  for (int i = 0; i < n_samples; ++i) {
    for (int j = 0; j < dim; ++j) {
      point[j] = dist(rng);
      out << point[j] << (j + 1 < dim ? "," : "");
    }
    out << "\n";
  }
}
