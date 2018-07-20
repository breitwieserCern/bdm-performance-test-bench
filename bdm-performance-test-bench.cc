#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <parallel/algorithm>

#include "param.h"
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
  Agent() : uuid_(counter_++) {
    for (uint64_t i = 0; i < 18; i++) {
      data_r_[i] = 1;
    }
    for (uint64_t i = 0; i < 18; i++) {
      data_w_[i] = 1;
    }
  }

  double Compute() {
    double sum = 0;
    for (int i = 0; i < 18; i++) {
      sum += data_r_[i];
      data_w_[i]++;
    }
    return sum / 18.0;
  }

  // not all data members of neighbors are accessed or updated
  double ComputeNeighbor() {
    double sum = 0;
    for (int i = 0; i < 9; i++) {
      sum += data_r_[i];
    }
    for (int i = 0; i < 6; i++) {
      data_w_[i]++;
    }
    return sum / 9.0;
  }

  Agent& operator+=(Agent& other) {
    // for (int i = 0; i < 9; i++) {
    //   data_r_[i] += other.data_r_[i];
    // }
    for (int i = 0; i < 6; i++) {
      data_w_[i] += other.data_w_[i];
    }
    return *this;
  }

  bool operator<(const Agent& other) {
    return uuid_ < other.uuid_;
  }

  friend bool operator<(const Agent& lhs, const Agent& rhs) {
    return lhs.uuid_ < rhs.uuid_;
  }

 private:
  static uint64_t counter_;
  uint64_t uuid_;
  double data_r_[18];
  double data_w_[18];
};

uint64_t Agent::counter_ = 0;

inline void FlushCache() {
  const uint64_t bigger_than_cachesize = 100 * 1024 * 1024;
  char* buffer = new char[bigger_than_cachesize];
  for (uint64_t i = 0; i < bigger_than_cachesize; i++) {
    buffer[i] = Param::num_agents_;
  }
  delete buffer;
}

enum NeighborMode { kConsecutive, kScattered };

static std::vector<int64_t> scattered;

inline uint64_t NeighborIndex(NeighborMode mode, uint64_t current_idx,
                              uint64_t num_neighbor) {
  if (mode == kConsecutive) {
    return std::min(Param::num_agents_ - 1, current_idx + num_neighbor + 1);
  } else if (mode == kScattered) {
    return std::min(Param::num_agents_ - 1,
                    current_idx + scattered[num_neighbor]);
    // return (current_idx * scattered[num_neighbor]) % Param::num_agents_;
  }
  throw false;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
double Classic(std::vector<Agent> agents, NeighborMode mode,
               TWorkload workload) {
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
  return total_sum;
}

// -----------------------------------------------------------------------------
void Sort(std::vector<Agent> agents) {

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(agents.begin(), agents.end(), g);
  Timer timer("sort    ");
  __gnu_parallel::sort(agents.begin(), agents.end());
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
double Patch(std::vector<Agent> agents, NeighborMode mode, TWorkload workload,
             uint64_t reuse) {
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
  return total_sum;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
inline void Run(NeighborMode mode, TWorkload workload) {
  std::vector<Agent> agents;
  agents.resize(Param::num_agents_);

  if (mode == kScattered) {
    scattered.resize(Param::neighbors_per_agent_);
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      int range = Param::neighbor_range_;
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
  Sort(agents);

  FlushCache();

  double expected =
      Param::num_agents_ *
      (1 + Param::neighbors_per_agent_ * Param::num_neighbor_ops_);
  EXPECT_NEAR(Classic(agents, mode, workload), expected);

  std::vector<uint64_t> reuse_vals = {0, 1, 2, 4, 8, 16, 32, 64};
  for (auto& r : reuse_vals) {
    FlushCache();
    EXPECT_NEAR(Patch(agents, mode, workload, r), expected);
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

  Agent a;
  std::cout << "Result for one agent: " << workload_per_cell(&a) << std::endl;

  PrintNewSection("strided access pattern");
  Run(kConsecutive, workload);

  PrintNewSection("scattered access pattern");
  Run(kScattered, workload);

  return 0;
}
