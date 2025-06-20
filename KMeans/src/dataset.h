#ifndef DATASET_GENERATOR_H
#define DATASET_GENERATOR_H

#include <string>

/// Generate `n_samples` points in `dim` dimensions,
/// with coordinates uniform in [0,10), and write to `filename` as CSV:
/// row = x₁,x₂,…,x_dim
void generate_dataset_csv(int n_samples, int dim, const std::string &filename);

#endif // DATASET_GENERATOR_H
