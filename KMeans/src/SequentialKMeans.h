// SequentialKMeans.h

#ifndef SEQUENTIAL_KMEANS_H
#define SEQUENTIAL_KMEANS_H

#include <vector>

class SequentialKMeans {
public:
  // Constructor
  SequentialKMeans(int k, int max_iters = 100, double tol = 1e-4);

  // Main fitting function
  void fit(const std::vector<std::vector<double>> &data);

  // Predict the cluster for a single new point
  int predict(const std::vector<double> &point) const;

  // --- GETTERS (ACCESSORS) ---

  // Get the final cluster centroids (already exists)
  std::vector<std::vector<double>> get_centroids() const;

  const std::vector<int> &get_assignments() const { return assignments_; }

private:
  // Parameters
  int k_;
  int max_iters_;
  double tol_;

  // Internal state
  size_t n_features_;
  std::vector<double> centroids_; // Flat vector for centroids
  std::vector<int> assignments_;

  // Helper methods (already exist)
  double squared_euclidean_distance(const double *a, const double *b) const;
  double euclidean_distance(const double *a, const double *b) const;
  int closest_centroid(const std::vector<double> &point) const;
  bool has_converged(const std::vector<double> &old_centroids_flat) const;
};

#endif // SEQUENTIAL_KMEANS_H
