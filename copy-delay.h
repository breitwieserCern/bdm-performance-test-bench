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
  Functor(Agent* neighbor) : neighbor_(neighbor) {}

  double operator()() { return neighbor_->ComputeNeighbor(); }
  Agent* neighbor_;
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
    // if(counter_ >= delayed_functions_.size()) {
    //   #pragma omp critical
    //   {
    //     if(counter_ < delayed_functions_.size()) {
    //       delayed_functions_.resize(delayed_functions_.size() * 1.5);
    //     }
    //   }
    // } else {
    //   delayed_functions_[counter_++] = f;
    // }

    // NB: not thread-safe: resize and assignment could happen at the same time
    // int idx = counter_++;
    // if (idx >= delayed_functions_.size()) {
    //   #pragma omp critical
    //   {
    //     if(idx < delayed_functions_.size()) {
    //       delayed_functions_.resize(delayed_functions_.size() * 1.5);
    //     }
    //   }
    // } else {
    //   delayed_functions_[idx] = f;
    // }

    std::lock_guard<std::mutex> lock(mutex_);
    delayed_functions_.emplace_back(std::move(f));
  }

  double Execute() {
    double sum = 0;
    // int size = counter_;
    // for (int i = 0; i < size; i++) {
    for (int i = 0; i < delayed_functions_.size(); i++) {
      sum += delayed_functions_[i]();
    }
    return sum;
  }

 private:
  std::mutex mutex_;
  // std::atomic<int> counter_;
  // std::vector<std::function<double()>> delayed_functions_;
  std::vector<Functor> delayed_functions_;
};

template <typename TWorkload, typename TWorkloadNeighbor>
void CopyDelay(NeighborMode mode, TWorkload workload_per_agent,
               TWorkloadNeighbor workload_neighbor, double expected) {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  std::vector<Agent> agents_t1 = Agent::Create(Param::num_agents_);
  std::vector<DelayedFunctions> delayed_functions;
  delayed_functions.resize(agents.size());

  auto for_each_neighbor = [&mode, &delayed_functions](
      uint64_t current_idx, std::vector<Agent>* agents,
      auto workload_neighbor) {
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      auto* neighbor = &((*agents)[nidx]);
      // delayed_functions[nidx].Delay([](){
      //   // return workload_neighbor(neighbor);
      // });
      delayed_functions[nidx].Delay(Functor(neighbor));
    }
  };

  auto workload = [&](auto for_each_neighbor, auto* agents, auto* current_agent,
                      uint64_t current_idx) {
    double sum = 0;
    sum += workload_per_agent(current_agent);
    for (uint64_t i = 0; i < Param::num_neighbor_ops_; i++) {
      for_each_neighbor(current_idx, agents, workload_neighbor);
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
    tl_sum += workload(for_each_neighbor, &agents, &copy, i);
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

  EXPECT_NEAR(total_sum, expected);
}

#endif  // COPY_DELAY_H_
