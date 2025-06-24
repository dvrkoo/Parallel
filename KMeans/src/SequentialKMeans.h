#ifndef SEQUENTIAL_KMEANS_H
#define SEQUENTIAL_KMEANS_H

#include <vector>

class SequentialKMeans {
public:
  SequentialKMeans(int k, int max_iters = 100, double tol = 1e-4);

  /**
   * @brief Fits the K-Means model to the given data.
   * @param data The input data, where each inner vector is a sample.
   */
  void fit(const std::vector<std::vector<double>> &data);

  /**
   * @brief Predicts the cluster for a single data point.
   * @param point The data point to predict.
   * @return The index of the closest cluster.
   */
  int predict(const std::vector<double> &point) const;

  /**
   * @brief Gets the final cluster centroids.
   * @return A vector of vectors, where each inner vector is a centroid.
   */
  std::vector<std::vector<double>> get_centroids() const;

private:
  // --- Parameters ---
  int k_;
  int max_iters_;
  double tol_;

  // --- Model State ---
  size_t n_features_;
  std::vector<double> centroids_;

  // --- Private Helper Methods ---
  double squared_euclidean_distance(const double *a, const double *b) const;
  double euclidean_distance(const double *a, const double *b) const;
  int closest_centroid(const std::vector<double> &point) const;
  bool has_converged(const std::vector<double> &old_centroids_flat) const;
};

#endif // SEQUENTIAL_KMEANS_H
