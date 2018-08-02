#ifndef COMMON_H_
#define COMMON_H_

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <mutex>
#include <parallel/algorithm>
#include <random>
#include <string>
#include <vector>

#include "param.h"
#include "timer.h"

// -----------------------------------------------------------------------------
#define EXPECT_NEAR(actual, expected)                                \
  if (std::fabs((expected) - (actual)) > 1e-5) {                     \
    std::cerr << "\033[1;31mWrong result on line: " << __LINE__      \
              << ", actual: " << actual << " expected: " << expected \
              << "\033[0m" << std::endl;                             \
  }

// -----------------------------------------------------------------------------
class SoaRefAgent;

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

  Agent(const Agent& other)
      : uuid_(other.uuid_), data_r_(other.data_r_), data_w_(other.data_w_) {}

  Agent(const SoaRefAgent&);

  Agent& operator=(const SoaRefAgent& other);

  uint32_t GetUuid() const { return uuid_; }

  double GetElement() const { return data_r_[0]; }

  double Compute() {
    double sum = 0;
    for (int i = 0; i < 18; i++) {
      if (Param::compute_intense_) {
        sum += std::exp(data_r_[i] - 1.0);
      } else {
        sum += data_r_[i];
      }
      data_w_[i]++;
    }
    return sum / 18.0;
  }

  // not all data members of neighbors are accessed or updated
  double ComputeNeighbor() {
    double sum = 0;
    for (int i = 0; i < 9; i++) {
      if (Param::compute_intense_) {
        sum += std::exp(data_r_[i] - 1.0);
      } else {
        sum += data_r_[i];
      }
    }
    for (int i = 0; i < 6; i++) {
      data_w_[i]++;
    }
    return sum / 18.0;
  }

  double ComputeNeighborReadPart() const {
    double sum = 0;
    for (int i = 0; i < 9; i++) {
      if (Param::compute_intense_) {
        sum += std::exp(data_r_[i] - 1.0);
      } else {
        sum += data_r_[i];
      }
    }
    return sum / 18.0;
  }

  void ComputeNeighborWritePart() {
    for (int i = 0; i < 6; i++) {
      data_w_[i]++;
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

  /// Calculate increment between the modified version and the reference and
  /// add it to this object.
  /// Can also be used to validate results: e.g. a reference must not be changed
  /// more than once in a single iteration.
  void ApplyDelta(const Agent& ref, const Agent& modified) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < 18; i++) {
      data_w_[i] += (modified.data_w_[i] - ref.data_w_[i]);
    }
  }

  Agent& operator=(const Agent& other) {
    uuid_ = other.uuid_;
    data_r_ = other.data_r_;
    data_w_ = other.data_w_;
    return *this;
  }

  bool operator<(const Agent& other) { return uuid_ < other.uuid_; }

  friend bool operator<(const Agent& lhs, const Agent& rhs) {
    return lhs.uuid_ < rhs.uuid_;
  }

  friend std::ostream& operator<<(std::ostream& str, const Agent& agent) {
    str << "{ uuid: " << agent.uuid_ << ", data_r_: [";
    for (auto& el : agent.data_r_) {
      str << el << ", ";
    }
    str << "], data_w_: [";
    for (auto& el : agent.data_w_) {
      str << el << ", ";
    }
    str << "]} ";
    return str;
  }

 private:
  friend SoaRefAgent;

  static uint32_t counter_;
  std::mutex mutex_;
  uint32_t uuid_;
  std::array<double, 18> data_r_;
  std::array<double, 18> data_w_;
};

uint32_t Agent::counter_ = 0;

// -----------------------------------------------------------------------------
// https://stackoverflow.com/questions/16465633/how-can-i-use-something-like-stdvectorstdmutex
struct mutex_wrapper : std::mutex {
  mutex_wrapper() = default;
  mutex_wrapper(mutex_wrapper const&) noexcept : std::mutex() {}
  bool operator==(mutex_wrapper const& other) noexcept {
    return this == &other;
  }
  mutex_wrapper& operator=(const mutex_wrapper&) { return *this; }
};

class SoaAgent {
 public:
  static SoaAgent Create(uint64_t elements) {
    SoaAgent agents;
    agents.mutex_.resize(elements);
    agents.uuid_.resize(elements);
    agents.data_r_.resize(elements);
    agents.data_w_.resize(elements);
    for (uint64_t i = 0; i < elements; i++) {
      agents.uuid_[i] = i;
      for (uint64_t j = 0; j < 18; j++) {
        agents.data_r_[i][j] = 1;
      }
      for (uint64_t j = 0; j < 18; j++) {
        agents.data_w_[i][j] = 0;
      }
    }
    return std::move(agents);
  }

  SoaAgent() {}

  SoaRefAgent operator[](uint64_t idx);

  uint64_t size() const { return uuid_.size(); }

  void clear() {
    mutex_.clear();
    uuid_.clear();
    data_r_.clear();
    data_w_.clear();
  }

  void resize(uint64_t elements) {
    mutex_.resize(elements);
    uuid_.resize(elements);
    data_r_.resize(elements);
    data_w_.resize(elements);
  }

  void Sort() {
    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(uuid_.begin(), uuid_.end(), g);

    std::vector<uint32_t> uuids = uuid_;

    decltype(data_r_) data_r_cpy;
    data_r_cpy.resize(data_r_.size());
    decltype(data_w_) data_w_cpy;
    data_w_cpy.resize(data_w_.size());

    Timer timer("sort SOA");
    // https://gcc.gnu.org/onlinedocs/libstdc++/manual/parallel_mode_using.html#parallel_mode.using.specific
    __gnu_parallel::sort(uuids.begin(), uuids.end());

#pragma omp parallel for
    for (uint64_t i = 0; i < size(); i++) {
      data_r_cpy[i] = data_r_[uuids[i]];
      data_w_cpy[i] = data_w_[uuids[i]];
    }

    data_r_ = std::move(data_r_cpy);
    data_w_ = std::move(data_w_cpy);
  }

 private:
  friend SoaRefAgent;

  std::vector<mutex_wrapper> mutex_;
  std::vector<uint32_t> uuid_;
  std::vector<std::array<double, 18>> data_r_;
  std::vector<std::array<double, 18>> data_w_;
};

class SoaRefAgent {
 public:
  SoaRefAgent(SoaAgent* agent, uint64_t idx)
      : kIdx(idx),
        mutex_(agent->mutex_),
        uuid_(agent->uuid_),
        data_r_(agent->data_r_),
        data_w_(agent->data_w_) {}

  SoaRefAgent& operator=(const Agent& agent) {
    uuid_[kIdx] = agent.uuid_;
    data_r_[kIdx] = agent.data_r_;
    data_w_[kIdx] = agent.data_w_;
    return *this;
  }

  SoaRefAgent& operator=(const SoaRefAgent& agent) {
    uuid_[kIdx] = agent.uuid_[agent.kIdx];
    data_r_[kIdx] = agent.data_r_[agent.kIdx];
    data_w_[kIdx] = agent.data_w_[agent.kIdx];
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& str, const SoaRefAgent& agent) {
    str << "{ uuid: " << agent.uuid_[agent.kIdx] << ", data_r_: [";
    for (auto& el : agent.data_r_[agent.kIdx]) {
      str << el << ", ";
    }
    str << "], data_w_: [";
    for (auto& el : agent.data_w_[agent.kIdx]) {
      str << el << ", ";
    }
    str << "]} ";
    return str;
  }

  uint32_t GetUuid() const { return uuid_[kIdx]; }

  double GetElement() const { return data_r_[kIdx][0]; }

  double Compute() {
    double sum = 0;
    for (int i = 0; i < 18; i++) {
      if (Param::compute_intense_) {
        sum += std::exp(data_r_[kIdx][i] - 1.0);
      } else {
        sum += data_r_[kIdx][i];
      }
      data_w_[kIdx][i]++;
    }
    return sum / 18.0;
  }

  // not all data members of neighbors are accessed or updated
  double ComputeNeighbor() {
    double sum = 0;
    for (int i = 0; i < 9; i++) {
      if (Param::compute_intense_) {
        sum += std::exp(data_r_[kIdx][i] - 1.0);
      } else {
        sum += data_r_[kIdx][i];
      }
    }
    for (int i = 0; i < 6; i++) {
      data_w_[kIdx][i]++;
    }
    return sum / 18.0;
  }

  double ComputeNeighborReadPart() const {
    double sum = 0;
    for (int i = 0; i < 9; i++) {
      sum += data_r_[kIdx][i];
    }
    return sum / 18.0;
  }

  void ComputeNeighborWritePart() {
    for (int i = 0; i < 6; i++) {
      data_w_[kIdx][i]++;
    }
  }

  double CheckSum() const {
    double sum = 0;
    for (int i = 0; i < 18; i++) {
      sum += data_r_[kIdx][i] * 3;
    }
    for (int i = 0; i < 18; i++) {
      sum += data_w_[kIdx][i] * 3;
    }
    return sum;
  }

  /// \see Agent::ApplyDelta
  void ApplyDelta(const SoaRefAgent& ref, const Agent& modified) {
    std::lock_guard<std::mutex> lock(mutex_[kIdx]);
    for (int i = 0; i < 18; i++) {
      data_w_[kIdx][i] += (modified.data_w_[i] - ref.data_w_[ref.kIdx][i]);
    }
  }

  /// \see Agent::ApplyDelta
  void ApplyDelta(const SoaRefAgent& ref, const SoaRefAgent& modified) {
    std::lock_guard<std::mutex> lock(mutex_[kIdx]);
    for (int i = 0; i < 18; i++) {
      data_w_[kIdx][i] +=
          (modified.data_w_[modified.kIdx][i] - ref.data_w_[ref.kIdx][i]);
    }
  }

  bool operator<(const SoaRefAgent& other) { return uuid_ < other.uuid_; }

  friend bool operator<(const SoaRefAgent& lhs, const SoaRefAgent& rhs) {
    return lhs.uuid_[lhs.kIdx] < rhs.uuid_[rhs.kIdx];
  }

 private:
  friend Agent;
  uint64_t kIdx;
  std::vector<mutex_wrapper>& mutex_;
  std::vector<uint32_t>& uuid_;
  std::vector<std::array<double, 18>>& data_r_;
  std::vector<std::array<double, 18>>& data_w_;
};

Agent::Agent(const SoaRefAgent& other)
    : uuid_(other.uuid_[other.kIdx]),
      data_r_(other.data_r_[other.kIdx]),
      data_w_(other.data_w_[other.kIdx]) {}

Agent& Agent::operator=(const SoaRefAgent& other) {
  uuid_ = other.uuid_[other.kIdx];
  data_r_ = other.data_r_[other.kIdx];
  data_w_ = other.data_w_[other.kIdx];
  return *this;
}

SoaRefAgent SoaAgent::operator[](uint64_t idx) {
  return SoaRefAgent(this, idx);
}

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
/// Generates neighbor index.
/// Used to test different memory access patterns.
/// NB: Neighbor relations are not symmetric (Neighbor(A): B != Neighbor(B): A)
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
    int64_t val = static_cast<int64_t>(r);
    // avoid neighbor index of 0:
    if (!val) {
      val++;
    }
    scattered[i] = val;
  }

  std::cout << std::endl << "memory offsets: " << std::endl;
  for (auto& el : scattered) {
    std::cout << el << ", ";
  }
  std::cout << std::endl << std::endl;
}

#endif  // COMMON_H_
