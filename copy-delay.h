#ifndef COPY_DELAY_H_
#define COPY_DELAY_H_

#include <atomic>
#include <mutex>

#include "common.h"
#include "timer.h"

// This scenario delays calls that mutate neighbors
// It is overly optimistic using Functor instead of a generic std::function

template <typename T, typename U, bool Expression>
struct ttop;

template <typename T, typename U>
struct ttop<T, U, true> {
  using type = T;
};

template <typename T, typename U>
struct ttop<T, U, false> {
  using type = U;
};

template <typename TAgent>
struct Functor {
  using TAgentRef = typename ttop<TAgent&, SoaRefAgent,
                                  std::is_same<TAgent, Agent>::value>::type;

  Functor() {}
  Functor(TAgentRef neighbor, bool mutate)
      : neighbor_(neighbor), mutate_(mutate) {}

  double operator()() {
    if (mutate_) {
      return neighbor_.ComputeNeighbor();
    } else {
      return neighbor_.ComputeNeighborReadPart();
    }
  }

  TAgentRef neighbor_;
  bool mutate_ = false;
};

template <typename TAgent>
class DelayedFunctions {
 public:
  DelayedFunctions() {
    // delayed_functions_.resize(Param::neighbors_per_agent_);
    delayed_functions_.reserve(Param::neighbors_per_agent_);
  }
  DelayedFunctions(const DelayedFunctions&) {}

  // void Delay(std::function<double()>&& f) {
  //   // std::lock_guard<std::mutex> lock(mutex);
  //   // delayed_functions_.emplace_back(std::move(f));
  //   if(counter_ > delayed_functions_.size()) {
  //     std::cout <<
  //     "-----------------------------------------------------------" <<
  //     std::endl;
  //   }
  //   delayed_functions_[counter_++] = f;
  // }

  void Delay(Functor<TAgent>&& f) {
    std::lock_guard<std::mutex> lock(mutex_);
    delayed_functions_.emplace_back(std::move(f));
  }

  double Execute() {
    double sum = 0;
    for (uint64_t i = 0; i < delayed_functions_.size(); i++) {
      sum += delayed_functions_[i]();
    }
    return sum;
  }

 private:
  std::mutex mutex_;
  // std::vector<std::function<double()>> delayed_functions_;
  std::vector<Functor<TAgent>> delayed_functions_;
};

template <typename TAgent>
void CopyDelay(NeighborMode mode, double expected) {
  auto&& agents = TAgent::Create(Param::num_agents_);
  auto&& agents_t1 = TAgent::Create(Param::num_agents_);
  std::vector<DelayedFunctions<TAgent>> delayed_functions;
  delayed_functions.resize(agents.size());

  auto for_each_neighbor = [&mode, &delayed_functions](uint64_t current_idx,
                                                       auto* agents) {
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      // delayed_functions[nidx].Delay([](){
      //   // return neighbor->ComputeNeighbor();
      // });
      bool mutate = i < Param::mutated_neighbors_;
      delayed_functions[nidx].Delay(Functor<TAgent>((*agents)[nidx], mutate));
    }
  };

  auto workload = [&](auto for_each_neighbor, auto* agents, auto* current_agent,
                      uint64_t current_idx) {
    double sum = 0;
    sum += current_agent->Compute();
    for (uint64_t i = 0; i < Param::num_neighbor_ops_; i++) {
      for_each_neighbor(current_idx, agents);
    }
    return sum;
  };

  FlushCache();

  thread_local double tl_sum = 0;
  thread_local Agent copy;
#pragma omp parallel
  tl_sum = 0;

  Timer timer("cpy-dly ");
#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    copy = agents[i];
    tl_sum += workload(for_each_neighbor, &agents_t1, &copy, i);
    agents_t1[i] = copy;
  }

#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    tl_sum += delayed_functions[i].Execute();
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }

  // check data member values
  double checksum = 0;
  for (uint64_t i = 0; i < agents.size(); i++) {
    checksum += agents_t1[i].CheckSum();
  }
  total_sum += checksum;

  EXPECT_NEAR(total_sum, expected);
}

#endif  // COPY_DELAY_H_
