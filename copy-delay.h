#ifndef COPY_DELAY_H_
#define COPY_DELAY_H_

#include <atomic>
#include <mutex>

#include "common.h"
#include "timer.h"

// This scenario delays calls that mutate neighbors
// It is overly optimistic using Functor instead of a generic std::function

struct Functor {
  Functor() {}
  Functor(Agent* neighbor, bool mutate)
      : neighbor_(neighbor), mutate_(mutate) {}

  double operator()() {
    if (mutate_) {
      return neighbor_->ComputeNeighbor();
    } else {
      return neighbor_->ComputeNeighborReadPart();
    }
  }

  Agent* neighbor_;
  bool mutate_ = false;
};

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

  void Delay(Functor&& f) {
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
  std::vector<Functor> delayed_functions_;
};

void CopyDelay(NeighborMode mode, double expected) {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  std::vector<Agent> agents_t1 = Agent::Create(Param::num_agents_);
  std::vector<DelayedFunctions> delayed_functions;
  delayed_functions.resize(agents.size());

  auto for_each_neighbor = [&mode, &delayed_functions](
      uint64_t current_idx, std::vector<Agent>* agents) {
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      auto* neighbor = &((*agents)[nidx]);
      // delayed_functions[nidx].Delay([](){
      //   // return neighbor->ComputeNeighbor();
      // });
      bool mutate = i < Param::mutated_neighbors_;
      delayed_functions[nidx].Delay(Functor(neighbor, mutate));
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
