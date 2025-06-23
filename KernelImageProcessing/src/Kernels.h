#pragma once
#include <vector>

namespace Kernels {
std::vector<float> PrewittX();    // 3x3 horizontal
std::vector<float> PrewittY();    // 3x3 horizontal
std::vector<float> Gaussian5x5(); // 5x5 blur
std::vector<float> Gaussian7x7(); // 7x7 blur
} // namespace Kernels
