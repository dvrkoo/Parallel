#include "ParallelKMeans.h"
#include "SequentialKMeans.h"
#include "dataset.h"

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

// Helper function to save a 2D vector (e.g., centroids) to a CSV file
void save_to_csv(const std::string &filename,
                 const std::vector<std::vector<double>> &data) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }
  file << std::fixed << std::setprecision(10);
  for (const auto &row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      file << row[i] << (i == row.size() - 1 ? "" : ",");
    }
    file << "\n";
  }
}

// Helper function to save a 1D vector (e.g., assignments) to a CSV file
void save_to_csv(const std::string &filename, const std::vector<int> &data) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << " for writing."
              << std::endl;
    return;
  }
  // Save each assignment on a new line
  for (const auto &val : data) {
    file << val << "\n";
  }
}

int main(int argc, char **argv) {
  int runs = 3; // how many repeats per configuration

  std::filesystem::create_directories("data");
  std::filesystem::create_directories("results");
  // Create a dedicated directory for plotting data
  std::filesystem::create_directories("results/plots");

  // The sample sizes to test
  std::vector<int> n_list = {1'0000};

  std::vector<int> k_list = {3};

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
      // We assume `fit` re-initializes state, so we can reuse the object.
      SequentialKMeans seq(k);
      for (int r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        seq.fit(data);
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_accum += std::chrono::duration<double>(t1 - t0).count();
      }
      double seq_mean = seq_accum / runs;

      // Save the results (centroids and assignments) from the last sequential
      {
        std::stringstream ss_centroids, ss_assignments;
        ss_centroids << "results/plots/centroids_seq_n" << n_samples << "_k"
                     << k << ".csv";
        ss_assignments << "results/plots/assignments_seq_n" << n_samples << "_k"
                       << k << ".csv";
        save_to_csv(ss_centroids.str(), seq.get_centroids());
        save_to_csv(ss_assignments.str(), seq.get_assignments());
      }

      // 3b) Parallel over thread counts
      for (int nt = 1; nt <= max_threads; ++nt) {
        double par_accum = 0.0;
        omp_set_num_threads(nt);
        ParallelKMeans par(k);
        for (int r = 0; r < runs; ++r) {
          auto tp0 = std::chrono::high_resolution_clock::now();
          par.fit(data);
          auto tp1 = std::chrono::high_resolution_clock::now();
          par_accum += std::chrono::duration<double>(tp1 - tp0).count();
        }
        double par_mean = par_accum / runs;
        double speedup = seq_mean / par_mean;

        // Save the results from the last parallel run for this configuration.
        {
          std::stringstream ss_centroids, ss_assignments;
          ss_centroids << "results/plots/centroids_par_n" << n_samples << "_k"
                       << k << "_t" << nt << ".csv";
          ss_assignments << "results/plots/assignments_par_n" << n_samples
                         << "_k" << k << "_t" << nt << ".csv";
          save_to_csv(ss_centroids.str(), par.get_centroids());
          save_to_csv(ss_assignments.str(), par.get_assignments());
        }

        // 4) Log to CSV
        out << n_samples << "," << k << "," << nt << "," << seq_mean << ","
            << par_mean << "," << speedup << "\n";
        out.flush();
      }

      std::cout << "Done: n=" << n_samples << ", k=" << k
                << " (seq=" << seq_mean << "s)\n";
    }
  }

  std::cout << "All results written to results/summary.csv\n";
  std::cout << "Clustering outputs for plotting saved in results/plots/\n";
  return 0;
}
