#include "ParallelKMeans.h"
#include "SequentialKMeans.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Helper to print centroids
auto print_centroids = [](const std::vector<std::vector<double>> &C,
                          const std::string &label) {
  std::cout << label << " centroids:\n";
  for (size_t i = 0; i < C.size(); ++i) {
    std::cout << "  [" << i << "]: ";
    for (double v : C[i])
      std::cout << std::fixed << std::setprecision(4) << v << " ";
    std::cout << "\n";
  }
};

// Compute max distance between two sets of centroids
auto max_centroid_diff = [](const std::vector<std::vector<double>> &A,
                            const std::vector<std::vector<double>> &B) {
  double maxdiff = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    for (size_t j = 0; j < A[i].size(); ++j)
      maxdiff = std::max(maxdiff, std::fabs(A[i][j] - B[i][j]));
  return maxdiff;
};

int main(int argc, char **argv) {
  // Read number of points and dimensionality
  int n_samples = 400000;
  int dim = 2;
  if (argc >= 3) {
    n_samples = std::stoi(argv[1]);
    dim = std::stoi(argv[2]);
  }
  std::cout << "Generating " << n_samples << " points in " << dim
            << " dimensions...\n";

  // Generate random data
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(0.0, 10.0);

  std::vector<std::vector<double>> data(n_samples, std::vector<double>(dim));
  for (int i = 0; i < n_samples; ++i)
    for (int j = 0; j < dim; ++j)
      data[i][j] = dist(rng);

  int k = 3;
  SequentialKMeans seq(k);
  ParallelKMeans par(k);

  // Time sequential fit
  auto t0 = std::chrono::high_resolution_clock::now();
  seq.fit(data);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur_seq = t1 - t0;

  // Time parallel fit
  auto t2 = std::chrono::high_resolution_clock::now();
  par.fit(data);
  auto t3 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> dur_par = t3 - t2;

  // Results
  auto Cseq = seq.get_centroids();
  auto Cpar = par.get_centroids();

  print_centroids(Cseq, "Sequential");
  print_centroids(Cpar, "Parallel");

  double maxDiff = max_centroid_diff(Cseq, Cpar);
  std::cout << "\nMax centroid difference: " << std::fixed
            << std::setprecision(6) << maxDiff
            << (maxDiff < 1e-6 ? " (OK)\n" : " (WARNING)\n");

  // Summarize match rate
  int match_count = 0;
  for (int i = 0; i < n_samples; ++i) {
    if (seq.predict(data[i]) == par.predict(data[i]))
      ++match_count;
  }
  double match_rate = 100.0 * match_count / n_samples;
  std::cout << "Match rate: " << match_count << "/" << n_samples << " ("
            << std::fixed << std::setprecision(2) << match_rate << "% )\n";

  // Timing summary
  std::cout << "\nTiming:\n"
            << std::fixed << std::setprecision(6)
            << "  Sequential: " << dur_seq.count() << " s\n"
            << "  Parallel:   " << dur_par.count() << " s\n"
            << std::setprecision(2)
            << "  Speedup:    " << (dur_seq.count() / dur_par.count()) << "x\n";

  return 0;
}
