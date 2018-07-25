// g++ -Wall -std=c++17 -g -O3 -fopenmp -D_GLIBCXX_PARALLEL
// bdm-performance-test-bench.cc -o bdm-performance-test-bench

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "common.h"
#include "copy-delay.h"
#include "copy.h"
#include "in-place.h"
#include "param.h"
#include "patch.h"
#include "sort.h"
#include "two_passes.h"

// -----------------------------------------------------------------------------
template <typename TWorkload>
inline void Run(NeighborMode mode, TWorkload workload) {

  double expected_checksum = Param::num_agents_ * Agent::ExpectedChecksum();

  double expected =
      Param::num_agents_ *
      (1 + Param::neighbors_per_agent_ * Param::num_neighbor_ops_ / 2.0) + expected_checksum;

  InPlace(mode, workload, expected);
  CopyDelay(mode, expected);
  Copy(mode, expected);
  TwoPasses(mode, expected);

  std::vector<uint64_t> reuse_vals = {0, 1, 2, 4, 8, 16, 32, 64};
  for (auto& r : reuse_vals) {
    Patch(mode, workload, r, expected);
  }
}

inline void PrintNewSection(const std::string& message) {
  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << message << std::endl << std::endl;
}

// -----------------------------------------------------------------------------
int main(int argc, const char** argv) {
  if (argc == 6) {
    Param::num_agents_ = std::atoi(argv[1]);
    Param::neighbors_per_agent_ = std::atoi(argv[2]);
    Param::mutated_neighbors_ = std::atoi(argv[3]);
    Param::num_neighbor_ops_ = std::atoi(argv[4]);
    Param::neighbor_range_ = std::atoi(argv[5]);
  } else if (argc != 1) {
    std::cout << "Wrong number of arguments!" << std::endl
              << "Usage: " << std::endl
              << "./bdm-performance-test-bench num_agents neighbors_per_agent mutated_neighbors_"
                 "num_neighbor_ops neighbor_range"
              << std::endl;
    return 1;
  }

  if (Param::mutated_neighbors_ > Param::neighbors_per_agent_) {
    std::cout << "ERROR: Parameter mutated_neighbors_ > neighbors_per_agent_" << std::endl;
    return 2;
  }

  auto workload = [&](auto for_each_neighbor, auto* agents,
                      uint64_t current_idx) {
    double sum = 0;
    Agent* current = &((*agents)[current_idx]);
    sum += current->Compute();
    for (uint64_t i = 0; i < Param::num_neighbor_ops_; i++) {
      sum += for_each_neighbor(current_idx, agents);
    }
    return sum;
  };

  Initialize();

  PrintNewSection("strided access pattern");
  Run(kConsecutive, workload);

  PrintNewSection("scattered access pattern");
  Run(kScattered, workload);

  PrintNewSection("sorting");
  Sort();
  SortMinCopies();

  return 0;
}
