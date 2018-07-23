// g++ -Wall -std=c++17 -g -O3 -fopenmp -D_GLIBCXX_PARALLEL bdm-performance-test-bench.cc -o bdm-performance-test-bench

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <parallel/algorithm>

#include "common.h"
#include "param.h"
#include "timer.h"


// -----------------------------------------------------------------------------
template <typename TWorkload>
void Classic(NeighborMode mode, TWorkload workload, double expected) {
  auto for_each_neighbor = [&mode](uint64_t current_idx,
                                   std::vector<Agent>* agents,
                                   auto workload_neighbor) {
    double sum = 0;
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      sum += workload_neighbor(&((*agents)[nidx]));
    }
    return sum;
  };

  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  FlushCache();

  thread_local double tl_sum = 0;
#pragma omp parallel
  tl_sum = 0;

  Timer timer("classic ");
#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    tl_sum += workload(for_each_neighbor, &agents, i);
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }

  EXPECT_NEAR(total_sum, expected);
}

// -----------------------------------------------------------------------------
void Sort() {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  FlushCache();

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(agents.begin(), agents.end(), g);
  Timer timer("sort    ");
  // https://gcc.gnu.org/onlinedocs/libstdc++/manual/parallel_mode_using.html#parallel_mode.using.specific
  __gnu_parallel::sort(agents.begin(), agents.end());
}

// -----------------------------------------------------------------------------
void SortMinCopies() {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  FlushCache();

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(agents.begin(), agents.end(), g);

  decltype(agents) sorted;
  sorted.resize(agents.size());

  std::vector<uint32_t> uuids;
  uuids.reserve(agents.size());
  for (uint64_t i = 0; i < agents.size(); i++) {
    uuids.push_back(agents[i].GetUuid() % agents.size());
  }

  Timer timer("sort MC ");
  // https://gcc.gnu.org/onlinedocs/libstdc++/manual/parallel_mode_using.html#parallel_mode.using.specific
  __gnu_parallel::sort(uuids.begin(), uuids.end());

  #pragma omp parallel for
  for(uint64_t i = 0; i < agents.size(); i++) {
    sorted[i] = agents[uuids[i]];
  }
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
void Patch(NeighborMode mode, TWorkload workload, uint64_t reuse, double expected) {
  const uint64_t num_agents = Param::num_agents_;

  auto for_each_neighbor = [](uint64_t current_idx, std::vector<Agent>* patch,
                              auto workload_per_cell) {
    double sum = 0;
    for (uint64_t i = 1; i < patch->size(); i++) {
      sum += workload_per_cell(&((*patch)[i]));
    }
    return sum;
  };

  auto add_neighbors_to_patch = [&mode](const auto& agents, auto* patch,
                                        uint64_t current_idx) {
    for (uint64_t i = 1; i < Param::neighbors_per_agent_ + 1; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      (*patch)[i] = agents[nidx];
    }
  };

  auto write_back_patch = [&mode](auto* agents, const auto& patch,
                                  uint64_t current_idx) {
    (*agents)[current_idx] = patch[0];
    for (uint64_t i = 1; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      (*agents)[nidx] = patch[i];
    }
  };

  std::vector<Agent> agents = Agent::Create(num_agents);
  FlushCache();

  thread_local std::vector<Agent> patch;
  thread_local std::vector<Agent> copy;
  thread_local std::vector<Agent> write_back_cache;
  thread_local double tl_sum = 0;

#pragma omp parallel
  {
    patch.resize(Param::neighbors_per_agent_ + 1);
    write_back_cache.clear();
    tl_sum = 0;
  }
  std::string padding = reuse < 10 ? " " : "";
  Timer timer(padding + std::to_string(reuse) + " Patch");
#pragma omp parallel for
  for (uint64_t i = 0; i < num_agents; i += (reuse + 1)) {
    patch[0] = agents[i];
    add_neighbors_to_patch(agents, &patch, i);

    if (reuse == 0) {
      tl_sum += workload(for_each_neighbor, &patch, 0);
      write_back_patch(&agents, patch, i);
    } else {
      copy = patch;
      for (uint64_t r = 0; r < reuse + 1 && r + i < num_agents; r++) {
        tl_sum += workload(for_each_neighbor, &copy, 0);
        write_back_cache = copy;
        // for (uint64_t el = 0; el < Param::neighbors_per_agent_; el++) {
        //   write_back_cache[el] += copy[el];
        // }
      }
      write_back_patch(&agents, write_back_cache, i);
    }
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }
  EXPECT_NEAR(total_sum, expected);
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
inline void Run(NeighborMode mode, TWorkload workload) {

  double expected =
      Param::num_agents_ *
      (1 + Param::neighbors_per_agent_ * Param::num_neighbor_ops_);

  Classic(mode, workload, expected);

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
  if (argc == 5) {
    Param::num_agents_ = std::atoi(argv[1]);
    Param::neighbors_per_agent_ = std::atoi(argv[2]);
    Param::num_neighbor_ops_ = std::atoi(argv[3]);
    Param::neighbor_range_ = std::atoi(argv[4]);
  } else if (argc != 1) {
    std::cout << "Wrong number of arguments!" << std::endl
              << "Usage: " << std::endl
              << "./bdm-performance-test-bench num_agents neighbors_per_agent "
                 "num_neighbor_ops neighbor_range"
              << std::endl;
    return 1;
  }

  auto workload_per_cell = [](Agent* current) {
    double sum = 0;
    sum += current->ComputeNeighbor();
    return sum;
  };

  auto workload_neighbor = [](Agent* current) {
    double sum = 0;
    sum += current->Compute();
    return sum;
  };

  auto workload = [&](auto for_each_neighbor, auto* agents,
                      uint64_t current_idx) {
    double sum = 0;
    Agent* current = &((*agents)[current_idx]);
    sum += workload_per_cell(current);
    for (uint64_t i = 0; i < Param::num_neighbor_ops_; i++) {
      sum += for_each_neighbor(current_idx, agents, workload_neighbor);
    }
    return sum;
  };

  Initialize();

  Agent a;
  std::cout << "Result for one agent: " << workload_per_cell(&a) << std::endl;

  PrintNewSection("strided access pattern");
  Run(kConsecutive, workload);

  PrintNewSection("scattered access pattern");
  Run(kScattered, workload);

  PrintNewSection("sorting");
  Sort();
  SortMinCopies();

  return 0;
}
