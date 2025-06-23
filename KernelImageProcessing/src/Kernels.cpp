#include "Kernels.h"

namespace Kernels {

std::vector<float> PrewittX() { return {-1, 0, 1, -1, 0, 1, -1, 0, 1}; }

std::vector<float> Gaussian5x5() {
  // Divide each by 256
  std::vector<int> k = {1,  4, 6, 4,  1,  4,  16, 24, 16, 4, 6, 24, 36,
                        24, 6, 4, 16, 24, 16, 4,  1,  4,  6, 4, 1};
  std::vector<float> out;
  out.reserve(25);
  for (int v : k)
    out.push_back(v / 256.f);
  return out;
}

std::vector<float> Gaussian7x7() {
  // Divide each by 1003
  std::vector<int> k = {0,  0, 1,  2,  1,  0,  0,  0,  3, 13, 22, 13,  3,
                        0,  1, 13, 59, 97, 59, 13, 1,  2, 22, 97, 159, 97,
                        22, 2, 1,  13, 59, 97, 59, 13, 1, 0,  3,  13,  22,
                        13, 3, 0,  0,  0,  1,  2,  1,  0, 0};
  std::vector<float> out;
  out.reserve(49);
  for (int v : k)
    out.push_back(v / 1003.f);
  return out;
}
} // namespace Kernels
