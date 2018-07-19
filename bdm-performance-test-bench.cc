#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "timer.h"

// -----------------------------------------------------------------------------
#define EXPECT_NEAR(expected, actual)                                        \
  if (std::fabs((expected) - (actual)) > 1e-5) {                             \
    std::cerr << "\033[1;31mWrong result on line: " << __LINE__ << "\033[0m" \
              << std::endl;                                                  \
  }

// -----------------------------------------------------------------------------
class Agent {
 public:
  Agent() {
    for (uint64_t i = 0; i < 36; i++) {
      data_[i] = 1;
    }
  }

  double Compute() {
    double sum = 0;
    for (int i = 0; i < 36; i++) {
      sum += data_[i];
    }
    return sum / 36.0;
  }

  // not all data members of are accessed or updated
  double ComputeNeighbor() {
    double sum = 0;
    for (int i = 0; i < 12; i++) {
      sum += data_[i];
    }
    // for (int i = 0; i < 6; i++) {
    //   sum += data_[i]++;
    // }
    return sum / 12.0;
  }

 private:
  double data_[36];
};

inline void FlushCache() {
  const uint64_t bigger_than_cachesize = 100 * 1024 * 1024;
  char* buffer = new char[bigger_than_cachesize];
  for (uint64_t i = 0; i < bigger_than_cachesize; i++) {
    buffer[i] = rand();
  }
  delete buffer;
}

enum NeighborMode { kConsecutive, kScattered };

static std::vector<int64_t> scattered;

inline uint64_t NeighborIndex(NeighborMode mode, uint64_t num_agents,
                              uint64_t current_idx, uint64_t num_neighbor) {
  if (mode == kConsecutive) {
    return std::min(num_agents - 1, current_idx + num_neighbor + 1);
  } else if (mode == kScattered) {
    return std::min(num_agents - 1, current_idx + scattered[num_neighbor]);
  }
  throw false;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
double Classic(std::vector<Agent> agents, NeighborMode mode,
               uint64_t neighbors_per_agent, TWorkload workload) {
  const uint64_t num_agents = agents.size();

  auto for_each_neighbor = [&num_agents, &neighbors_per_agent, &mode](
      uint64_t current_idx, std::vector<Agent>* agents,
      auto workload_neighbor) {
    double sum = 0;
    for (uint64_t i = 0; i < neighbors_per_agent; i++) {
      uint64_t nidx = NeighborIndex(mode, num_agents, current_idx, i);
      sum += workload_neighbor(&((*agents)[nidx]));
    }
    return sum;
  };

  thread_local double tl_sum = 0;
#pragma omp parallel
  tl_sum = 0;

  Timer timer("classic");
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
  std::cout << "    result: " << total_sum << std::endl;
  return total_sum;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
double Patch(std::vector<Agent> agents, NeighborMode mode,
             uint64_t neighbors_per_agent, TWorkload workload, uint64_t reuse) {
  const uint64_t num_agents = agents.size();

  auto for_each_neighbor = [](uint64_t current_idx, std::vector<Agent>* patch,
                              auto workload_per_cell) {
    double sum = 0;
    for (uint64_t i = 1; i < patch->size(); i++) {
      sum += workload_per_cell(&((*patch)[i]));
    }
    return sum;
  };

  auto add_neighbors_to_patch = [&mode](const auto& agents, auto* patch,
                                        uint64_t neighbors_per_agent,
                                        uint64_t current_idx) {
    uint64_t num_agents = agents.size();
    for (uint64_t i = 0; i < neighbors_per_agent; i++) {
      uint64_t nidx = NeighborIndex(mode, num_agents, current_idx, i);
      patch->push_back(agents[nidx]);
    }
  };

  auto write_back_patch = [&mode](auto* agents, const auto& patch,
                                  uint64_t neighbors_per_agent,
                                  uint64_t current_idx) {
    (*agents)[current_idx] = patch[0];
    uint64_t num_agents = agents->size();
    for (uint64_t i = 1; i < neighbors_per_agent; i++) {
      uint64_t nidx = NeighborIndex(mode, num_agents, current_idx, i);
      (*agents)[nidx] = patch[i];
    }
  };

  thread_local std::vector<Agent> patch;
  thread_local std::vector<Agent> copy;
  thread_local std::vector<Agent> write_back_cache;
  thread_local double tl_sum = 0;

#pragma omp parallel
  {
    patch.reserve(neighbors_per_agent);
    write_back_cache.clear();
    tl_sum = 0;
  }

  Timer timer("Patch  ");
#pragma omp parallel for
  for (uint64_t i = 0; i < num_agents; i += (reuse + 1)) {
    patch.clear();
    patch.push_back(agents[i]);
    add_neighbors_to_patch(agents, &patch, neighbors_per_agent, i);

    if (reuse == 0) {
      tl_sum += workload(for_each_neighbor, &patch, 0);
      write_back_patch(&agents, patch, neighbors_per_agent, i);
    } else {
      copy = patch;
      for (uint64_t r = 0; r < reuse + 1 && r + i < num_agents; r++) {
        tl_sum += workload(for_each_neighbor, &copy, 0);
        write_back_cache = copy;
        // for (int el = 0; el < neighbors_per_agent; el++) {
        //   write_back_cache[el] += copy[el];
        // }
      }
      write_back_patch(&agents, write_back_cache, neighbors_per_agent, i);
    }
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }
  std::cout << "    result: " << total_sum << std::endl;
  std::cout << "    reuse : " << reuse << std::endl;
  return total_sum;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
inline void Run(uint64_t num_agents, uint64_t neighbors_per_agent,
                NeighborMode mode, TWorkload workload) {
  std::vector<Agent> agents;
  agents.resize(num_agents);

  if (mode == kScattered) {
    scattered.resize(neighbors_per_agent);
    for (uint64_t i = 0; i < neighbors_per_agent; i++) {
      int range = 1e5;
      double r = (rand() / static_cast<double>(RAND_MAX) - 0.5) *
                 range;  // r (-range, range)
      scattered[i] = static_cast<int64_t>(r);
    }
    std::cout << std::endl << "memory offsets: " << std::endl;
    for (auto& el : scattered) {
      std::cout << el << ", ";
    }
    std::cout << std::endl << std::endl;
  }

  FlushCache();

  double expected = num_agents + num_agents * neighbors_per_agent;
  EXPECT_NEAR(Classic(agents, mode, neighbors_per_agent, workload), expected);

  std::vector<uint64_t> reuse_vals = {0, 1, 2, 4, 8, 16, 32, 64};
  for (auto& r : reuse_vals) {
    FlushCache();
    EXPECT_NEAR(Patch(agents, mode, neighbors_per_agent, workload, r),
                expected);
  }
}

inline void PrintNewSection(const std::string& message) {
  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << message << std::endl << std::endl;
}

// -----------------------------------------------------------------------------
int main(int argc, const char** argv) {
  uint64_t num_agents = 1e7;
  uint64_t neighbors_per_agent = 9;

  if (argc == 3) {
    num_agents = std::atoi(argv[1]);
    neighbors_per_agent = std::atoi(argv[2]);
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

  auto workload = [&](auto for_each_neighbor, std::vector<Agent>* agents,
                      uint64_t current_idx) {
    double sum = 0;
    Agent* current = &((*agents)[current_idx]);
    sum += workload_per_cell(current);
    sum += for_each_neighbor(current_idx, agents, workload_neighbor);
    return sum;
  };

  Agent a;
  std::cout << "Result for one agent: " << workload_per_cell(&a) << std::endl;

  PrintNewSection("strided access pattern");
  Run(num_agents, neighbors_per_agent, kConsecutive, workload);

  PrintNewSection("scattered access pattern");
  Run(num_agents, neighbors_per_agent, kScattered, workload);

  return 0;
}
