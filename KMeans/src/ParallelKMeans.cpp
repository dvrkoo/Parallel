#include "ParallelKMeans.h"
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <random>

ParallelKMeans::ParallelKMeans(int k, int max_iters, double tol)
    : k(k), max_iters(max_iters), tol(tol) {}

void ParallelKMeans::fit(const std::vector<std::vector<double>> &data) {
  int n_samples = data.size();
  int n_features = data[0].size();
  // Set random seed for reproducibility
  std::mt19937 rng(12345);
  std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
  // Initialize centroids randomly
  centroids =
      std::vector<std::vector<double>>(k, std::vector<double>(n_features));
  for (int i = 0; i < k; ++i) {
    // Choose a not so random sample from the data
    centroids[i] = data[dist(rng)];
  }

  std::vector<int> labels(n_samples);

  for (int iter = 0; iter < max_iters; ++iter) {
// ==== Assignment step (Parallel) ====
#pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
      labels[i] = closest_centroid(data[i]);
    }

    // Save old centroids for convergence check
    std::vector<std::vector<double>> old_centroids = centroids;

    // ==== Update step (Parallel Reduction) ====
    std::vector<std::vector<double>> new_centroids(
        k, std::vector<double>(n_features, 0.0));
    std::vector<int> counts(k, 0);

#pragma omp parallel
    {
      std::vector<std::vector<double>> local_sums(
          k, std::vector<double>(n_features, 0.0));
      std::vector<int> local_counts(k, 0);

#pragma omp for nowait
      for (int i = 0; i < n_samples; ++i) {
        int cluster = labels[i];
        for (int j = 0; j < n_features; ++j) {
          local_sums[cluster][j] += data[i][j];
        }
        local_counts[cluster]++;
      }

// Merge local sums into global sums
#pragma omp critical
      {
        for (int c = 0; c < k; ++c) {
          for (int j = 0; j < n_features; ++j) {
            new_centroids[c][j] += local_sums[c][j];
          }
          counts[c] += local_counts[c];
        }
      }
    }

    // Finalize centroids
    for (int c = 0; c < k; ++c) {
      if (counts[c] == 0)
        continue;
      for (int j = 0; j < n_features; ++j) {
        new_centroids[c][j] /= counts[c];
      }
    }

    centroids = new_centroids;

    if (has_converged(old_centroids))
      break;
  }
}

int ParallelKMeans::predict(const std::vector<double> &point) const {
  return closest_centroid(point);
}

const std::vector<std::vector<double>> &ParallelKMeans::get_centroids() const {
  return centroids;
}

double ParallelKMeans::euclidean_distance(const std::vector<double> &a,
                                          const std::vector<double> &b) const {
  double sum = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

int ParallelKMeans::closest_centroid(const std::vector<double> &point) const {
  double min_dist = std::numeric_limits<double>::max();
  int label = -1;
  for (int i = 0; i < k; ++i) {
    double dist = euclidean_distance(point, centroids[i]);
    if (dist < min_dist) {
      min_dist = dist;
      label = i;
    }
  }
  return label;
}

bool ParallelKMeans::has_converged(
    const std::vector<std::vector<double>> &old_centroids) const {
  for (int i = 0; i < k; ++i) {
    if (euclidean_distance(old_centroids[i], centroids[i]) > tol) {
      return false;
    }
  }
  return true;
}
