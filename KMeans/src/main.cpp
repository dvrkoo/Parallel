#include "ParallelKMeans.h"
#include "SequentialKMeans.h"
#include "dataset.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

// Load a CSV (skipping header) into vector<vector<double>>
std::vector<std::vector<double>> load_csv(const std::string &fname) {
  std::ifstream in(fname);
  std::string line;
  std::vector<std::vector<double>> data;
  std::getline(in, line); // skip header
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::vector<double> row;
    double val;
    while (ss >> val) {
      row.push_back(val);
      if (ss.peek() == ',')
        ss.ignore();
    }
    data.push_back(row);
  }
  return data;
}

int main(int argc, char **argv) {
  int runs = 3; // how many repeats per configuration

  std::filesystem::create_directories("data");
  std::filesystem::create_directories("results");

  // The sample sizes to test
  std::vector<int> n_list = {100,     1'000,     10'000,
                             100'000, 1'000'000, 10'000'000};

  std::vector<int> k_list = {3, 5, 10, 15, 20, 25, 30, 40, 50};

  // Cap threads to 12 (Ryzen 5 3600 has 12 logical threads)
  int max_threads = std::min(omp_get_max_threads(), 12);

  // Prepare output CSV
  std::ofstream out("results/summary.csv");
  out << "n_samples,k,threads,seq_time,tp_time,speedup\n";
  out << std::fixed << std::setprecision(6);

  // RNG for dataset generation is fixed inside dataset_generator

  for (int n_samples : n_list) {
    // 1) Generate or skip dataset
    std::string datafile =
        "data/" + std::to_string(n_samples) + "_" + std::to_string(3) + ".csv";
    if (!std::filesystem::exists(datafile)) {
      std::cout << "Generating " << datafile << "...\n";
      generate_dataset_csv(n_samples, 3, datafile);
    }

    // 2) Load data once
    auto data = load_csv(datafile);

    // 3) Loop over k
    for (int k : k_list) {
      // 3a) Sequential baseline (mean over runs)
      double seq_accum = 0.0;
      for (int r = 0; r < runs; ++r) {
        SequentialKMeans seq(k);
        auto t0 = std::chrono::high_resolution_clock::now();
        seq.fit(data);
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_accum += std::chrono::duration<double>(t1 - t0).count();
      }
      double seq_mean = seq_accum / runs;

      // 3b) Parallel over thread counts
      for (int nt = 1; nt <= max_threads; ++nt) {
        double par_accum = 0.0;
        omp_set_num_threads(nt);
        for (int r = 0; r < runs; ++r) {
          ParallelKMeans par(k);
          auto tp0 = std::chrono::high_resolution_clock::now();
          par.fit(data);
          auto tp1 = std::chrono::high_resolution_clock::now();
          par_accum += std::chrono::duration<double>(tp1 - tp0).count();
        }
        double par_mean = par_accum / runs;
        double speedup = seq_mean / par_mean;

        // 4) Log to CSV
        out << n_samples << "," << k << "," << nt << "," << seq_mean << ","
            << par_mean << "," << speedup << "\n";
      }

      std::cout << "Done: n=" << n_samples << ", k=" << k
                << " (seq=" << seq_mean << "s)\n";
    }
  }

  std::cout << "All results written to results/summary.csv\n";
  return 0;
}
