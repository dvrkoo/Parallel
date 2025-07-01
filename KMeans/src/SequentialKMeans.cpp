#include "SequentialKMeans.h"
#include <algorithm> // For std::copy
#include <cmath>
#include <cstdlib>
#include <limits>
#include <random>

SequentialKMeans::SequentialKMeans(int k, int max_iters, double tol)
    : k_(k), max_iters_(max_iters), tol_(tol), n_features_(0) {}

void SequentialKMeans::fit(const std::vector<std::vector<double>> &data) {
  if (data.empty()) {
    return;
  }
  size_t n_samples = data.size();
  n_features_ = data[0].size();

  // --- OPTIMIZATION 1: Flatten input data ---
  std::vector<double> data_flat(n_samples * n_features_);
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

  // Use a local variable for labels within the loop
  std::vector<int> labels(n_samples);

  for (int iter = 0; iter < max_iters_; ++iter) {
    // ==== Assignment step ====
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
      labels[i] = best_label;
    }

    std::vector<double> old_centroids_flat = centroids_;

    // ==== Update step ====
    std::vector<double> new_centroids_flat(k_ * n_features_, 0.0);
    std::vector<int> counts(k_, 0);

    for (size_t i = 0; i < n_samples; ++i) {
      int cluster = labels[i];
      if (cluster != -1) {
        counts[cluster]++;
        for (size_t j = 0; j < n_features_; ++j) {
          new_centroids_flat[cluster * n_features_ + j] +=
              data_flat[i * n_features_ + j];
        }
      }
    }

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

  // **CHANGE**: After the loop, run the assignment step one final time to
  // ensure the assignments correspond to the *final* centroids.
  assignments_.resize(n_samples);
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
    // Store the final assignment in the member variable
    assignments_[i] = best_label;
  }
}

// ========= NO CHANGES NEEDED FOR THE METHODS BELOW =========

int SequentialKMeans::predict(const std::vector<double> &point) const {
  return closest_centroid(point);
}

std::vector<std::vector<double>> SequentialKMeans::get_centroids() const {
  std::vector<std::vector<double>> result(k_, std::vector<double>(n_features_));
  for (int i = 0; i < k_; ++i) {
    for (size_t j = 0; j < n_features_; ++j) {
      result[i][j] = centroids_[i * n_features_ + j];
    }
  }
  return result;
}

double SequentialKMeans::squared_euclidean_distance(const double *a,
                                                    const double *b) const {
  double sum = 0.0;
  for (size_t i = 0; i < n_features_; ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

double SequentialKMeans::euclidean_distance(const double *a,
                                            const double *b) const {
  return std::sqrt(squared_euclidean_distance(a, b));
}

int SequentialKMeans::closest_centroid(const std::vector<double> &point) const {
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

bool SequentialKMeans::has_converged(
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
