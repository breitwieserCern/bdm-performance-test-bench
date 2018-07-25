#ifndef COMMON_H_
#define COMMON_H_

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "param.h"

// -----------------------------------------------------------------------------
#define EXPECT_NEAR(expected, actual)                                        \
  if (std::fabs((expected) - (actual)) > 1e-5) {                             \
    std::cerr << "\033[1;31mWrong result on line: " << __LINE__ << "\033[0m" \
              << std::endl;                                                  \
  }

// -----------------------------------------------------------------------------
class Agent {
 public:
  static std::vector<Agent> Create(uint64_t elements) {
    std::vector<Agent> agents;
    agents.resize(elements);
    return std::move(agents);
  }

  static double ExpectedChecksum() {
    return 3 * (18 + 6 * Param::mutated_neighbors_ + 18);
  }

  Agent() : uuid_(counter_++) {
    for (uint64_t i = 0; i < 18; i++) {
      data_r_[i] = 1;
    }
    for (uint64_t i = 0; i < 18; i++) {
      data_w_[i] = 0.0;
    }
  }

  Agent(const Agent& other) : uuid_(other.uuid_) {
    for (int i = 0; i < 18; i++) {
      data_r_[i] = other.data_r_[i];
    }
    for (int i = 0; i < 18; i++) {
      data_w_[i] = other.data_w_[i];
    }
  }

  uint32_t GetUuid() const { return uuid_; }

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
    return sum / 18.0;
  }

  double ComputeNeighborReadPart() const {
    double sum = 0;
    for (int i = 0; i < 9; i++) {
      sum += data_r_[i];
    }
    return sum / 18.0;
  }

  void ComputeNeighborWritePart(double increment) {
    for (int i = 0; i < 6; i++) {
      data_w_[i] += increment;
    }
  }

  double CheckSum() const {
    double sum = 0;
    for (int i = 0; i < 18; i++) {
      sum += data_r_[i] * 3;
    }
    for (int i = 0; i < 18; i++) {
      sum += data_w_[i] * 3;
    }
    return sum;
  }

  Agent& operator+=(const Agent& other) {
    // for (int i = 0; i < 9; i++) {
    //   data_r_[i] += other.data_r_[i];
    // }
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < 18; i++) {
      data_w_[i] += other.data_w_[i];
    }
    return *this;
  }

  Agent& operator=(const Agent& other) {
    uuid_ = other.uuid_;
    for (int i = 0; i < 18; i++) {
      data_r_[i] = other.data_r_[i];
    }
    for (int i = 0; i < 18; i++) {
      data_w_[i] = other.data_w_[i];
    }
    return *this;
  }

  bool operator<(const Agent& other) { return uuid_ < other.uuid_; }

  friend bool operator<(const Agent& lhs, const Agent& rhs) {
    return lhs.uuid_ < rhs.uuid_;
  }

 private:
  static uint32_t counter_;
  std::mutex mutex_;
  uint32_t uuid_;
  double data_r_[18];
  double data_w_[18];
};

uint32_t Agent::counter_ = 0;

// -----------------------------------------------------------------------------
inline void FlushCache() {
  const uint64_t bigger_than_cachesize = 100 * 1024 * 1024;
  char* buffer = new char[bigger_than_cachesize];
  char r = rand();
  for (uint64_t i = 0; i < bigger_than_cachesize; i++) {
    buffer[i] = r + i;
  }
  delete buffer;
}

enum NeighborMode { kConsecutive, kScattered };

static std::vector<int64_t> scattered;

// -----------------------------------------------------------------------------
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
inline void Initialize() {
  scattered.resize(Param::neighbors_per_agent_);
  for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
    int range = Param::neighbor_range_;
    // r (-range, range)
    double r = (rand() / static_cast<double>(RAND_MAX) - 0.5) * range;
    scattered[i] = static_cast<int64_t>(r);
  }
  std::cout << std::endl << "memory offsets: " << std::endl;
  for (auto& el : scattered) {
    std::cout << el << ", ";
  }
  std::cout << std::endl << std::endl;
}

#endif  // COMMON_H_
