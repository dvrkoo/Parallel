#pragma once
#include <array>
#include <vector>

namespace Kernels {
std::vector<float> PrewittX();    // 3x3 horizontal
std::vector<float> PrewittY();    // 3x3 horizontal
std::vector<float> Gaussian5x5(); // 5x5 blur
std::vector<float> Gaussian7x7(); // 7x7 blur
// A 5x5 Laplacian of Gaussian (LoG) kernel
inline std::array<float, 25> LaplacianOfGaussian5x5() {
  return {0,  0,  -1, 0,  0,  0,  -1, -2, -1, 0,  -1, -2, 16,
          -2, -1, 0,  -1, -2, -1, 0,  0,  0,  -1, 0,  0};
}

// A 3x3 Sharpen kernel
inline std::array<float, 9> Sharpen() {
  return {0, -1, 0, -1, 5, -1, 0, -1, 0};
}
} // namespace Kernels
