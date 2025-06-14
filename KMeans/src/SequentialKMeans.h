
#ifndef SEQUENTIAL_KMEANS_H
#define SEQUENTIAL_KMEANS_H

#include <vector>

class SequentialKMeans {
public:
  SequentialKMeans(int k, int max_iters = 100, double tol = 1e-4);

  void fit(const std::vector<std::vector<double>> &data);
  int predict(const std::vector<double> &point) const;
  const std::vector<std::vector<double>> &get_centroids() const;

private:
  int k;
  int max_iters;
  double tol;
  std::vector<std::vector<double>> centroids;

  double euclidean_distance(const std::vector<double> &a,
                            const std::vector<double> &b) const;
  int closest_centroid(const std::vector<double> &point) const;
  bool
  has_converged(const std::vector<std::vector<double>> &old_centroids) const;
};

#endif // KMEANS_H
