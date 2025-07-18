#include "ParallelKMeans.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <random>

ParallelKMeans::ParallelKMeans(int k, int max_iters, double tol)
    : k_(k), max_iters_(max_iters), tol_(tol), n_features_(0) {}

void ParallelKMeans::fit(const std::vector<std::vector<double>> &data) {
  if (data.empty()) {
    return;
  }
  size_t n_samples = data.size();
  n_features_ = data[0].size();

  // Resize the member variable to hold assignments for the current dataset
  assignments_.assign(n_samples, -1); // Initialize all assignments to -1

  std::vector<double> data_flat(n_samples * n_features_);
#pragma omp parallel for
  for (size_t i = 0; i < n_samples; ++i) {
    for (size_t j = 0; j < n_features_; ++j) {
      data_flat[i * n_features_ + j] = data[i][j];
    }
  }

  // --- Initialize centroids randomly ---
  centroids_.resize(k_ * n_features_);
  std::mt19937 rng(12345); // For reproducibility
  std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
  for (int i = 0; i < k_; ++i) {
    size_t random_sample_idx = dist(rng);
    const double *sample_start = &data_flat[random_sample_idx * n_features_];
    double *centroid_start = &centroids_[i * n_features_];
    std::copy(sample_start, sample_start + n_features_, centroid_start);
  }

  for (int iter = 0; iter < max_iters_; ++iter) {
    // ==== Assignment step (Parallel) ====
#pragma omp parallel for
    for (size_t i = 0; i < n_samples; ++i) {
      const double *point = &data_flat[i * n_features_];
      double min_dist_sq = std::numeric_limits<double>::max();
      int best_label = -1;
      for (int c = 0; c < k_; ++c) {
        const double *centroid = &centroids_[c * n_features_];
        double dist_sq = squared_euclidean_distance(point, centroid);
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          best_label = c;
        }
      }
      assignments_[i] = best_label;
    }

    // Save old centroids for convergence check
    std::vector<double> old_centroids_flat = centroids_;

    // ==== Update step (Efficient Parallel Reduction) ====
    std::vector<double> new_centroids_flat(k_ * n_features_, 0.0);
    std::vector<int> counts(k_, 0);

    auto *new_centroids_ptr = new_centroids_flat.data();
    auto *counts_ptr = counts.data();
#pragma omp parallel for reduction(+ : new_centroids_ptr[ : k_ * n_features_]) \
    reduction(+ : counts_ptr[ : k_])
    for (size_t i = 0; i < n_samples; ++i) {
      int cluster = assignments_[i];
      if (cluster != -1) {
        counts_ptr[cluster]++;
        for (size_t j = 0; j < n_features_; ++j) {
          new_centroids_ptr[cluster * n_features_ + j] +=
              data_flat[i * n_features_ + j];
        }
      }
    }

    // Finalize centroids by dividing by counts
#pragma omp parallel for
    for (int c = 0; c < k_; ++c) {
      if (counts[c] > 0) {
        for (size_t j = 0; j < n_features_; ++j) {
          new_centroids_flat[c * n_features_ + j] /= counts[c];
        }
      }
    }

    centroids_ = new_centroids_flat;

    if (has_converged(old_centroids_flat)) {
      break;
    }
  }
}

int ParallelKMeans::predict(const std::vector<double> &point) const {
  return closest_centroid(point);
}

std::vector<std::vector<double>> ParallelKMeans::get_centroids() const {
  std::vector<std::vector<double>> result(k_, std::vector<double>(n_features_));
  for (int i = 0; i < k_; ++i) {
    for (size_t j = 0; j < n_features_; ++j) {
      result[i][j] = centroids_[i * n_features_ + j];
    }
  }
  return result;
}

double ParallelKMeans::squared_euclidean_distance(const double *a,
                                                  const double *b) const {
  double sum = 0.0;
  for (size_t i = 0; i < n_features_; ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

double ParallelKMeans::euclidean_distance(const double *a,
                                          const double *b) const {
  return std::sqrt(squared_euclidean_distance(a, b));
}

int ParallelKMeans::closest_centroid(const std::vector<double> &point) const {
  if (centroids_.empty())
    return -1;

  double min_dist_sq = std::numeric_limits<double>::max();
  int label = -1;
  const double *point_ptr = point.data();

  for (int i = 0; i < k_; ++i) {
    const double *centroid_ptr = &centroids_[i * n_features_];
    double dist_sq = squared_euclidean_distance(point_ptr, centroid_ptr);
    if (dist_sq < min_dist_sq) {
      min_dist_sq = dist_sq;
      label = i;
    }
  }
  return label;
}

bool ParallelKMeans::has_converged(
    const std::vector<double> &old_centroids_flat) const {
  for (int i = 0; i < k_; ++i) {
    const double *old_c = &old_centroids_flat[i * n_features_];
    const double *new_c = &centroids_[i * n_features_];
    if (euclidean_distance(old_c, new_c) > tol_) {
      return false;
    }
  }
  return true;
}
