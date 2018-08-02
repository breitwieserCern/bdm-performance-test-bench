#ifndef COPY_DELAY_H_
#define COPY_DELAY_H_

#include <atomic>
#include <functional>
#include <mutex>

#include "common.h"
#include "timer.h"

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
  DelayedFunctions() {}
  DelayedFunctions(const DelayedFunctions&) {}

  void Initialize(bool use_function) {
    use_function_ = use_function;
    if (use_function_) {
      delayed_functions_.reserve(Param::neighbors_per_agent_);
    } else {
      delayed_functors_.reserve(Param::neighbors_per_agent_);
    }
  }

  void Delay(Functor<TAgent>&& f) {
    std::lock_guard<std::mutex> lock(mutex_);
    delayed_functors_.emplace_back(std::move(f));
  }

  void Delay(std::function<double()>&& f) {
    std::lock_guard<std::mutex> lock(mutex_);
    delayed_functions_.emplace_back(std::move(f));
  }

  double Execute() {
    double sum = 0;
    if (use_function_) {
      for (uint64_t i = 0; i < delayed_functions_.size(); i++) {
        sum += delayed_functions_[i]();
      }
    } else {
      for (uint64_t i = 0; i < delayed_functors_.size(); i++) {
        sum += delayed_functors_[i]();
      }
    }
    return sum;
  }

 private:
  std::mutex mutex_;
  bool use_function_;
  std::vector<std::function<double()>> delayed_functions_;
  std::vector<Functor<TAgent>> delayed_functors_;
};

template <typename TAgent>
void DelayInternal(NeighborMode mode, double expected, bool use_function) {
  auto&& agents = TAgent::Create(Param::num_agents_);
  auto&& agents_t1 = TAgent::Create(Param::num_agents_);

  std::vector<DelayedFunctions<TAgent>> delayed_functions;
  delayed_functions.resize(agents.size());
  for (auto& el : delayed_functions) {
    el.Initialize(use_function);
  }

  auto for_each_neighbor = [&mode, &delayed_functions, use_function](
      uint64_t current_idx, auto* agents) {
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      bool mutate = i < Param::mutated_neighbors_;
      if (use_function) {
        delayed_functions[nidx].Delay([=]() {
          if (mutate) {
            return (*agents)[nidx].ComputeNeighbor();
          } else {
            return (*agents)[nidx].ComputeNeighborReadPart();
          }
        });
      } else {
        delayed_functions[nidx].Delay(Functor<TAgent>((*agents)[nidx], mutate));
      }
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

  std::string timer_name;
  if (use_function) {
    timer_name = "delay 0 ";
  } else {
    timer_name = "delay 1 ";
  }
  Timer timer(timer_name);
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

/// This scenario delays calls that mutate neighbors.
/// It adds them to a list of delayed functions of the neighbor. Therefore,
/// neighbors are still modified, but the enqueuing is protected by a lock.
/// Runs two versions: using a Functor and a std::function.
/// Using a functor is overly optimistic. It is only able to delay one type of
/// call and it doesn't require a heap allocation.
/// Assumes that calling non-const functions on neighbors can be delayed.
/// Discretization issue is solved by copying the current agent.
template <typename TAgent>
void Delay(NeighborMode mode, double expected) {
  DelayInternal<TAgent>(mode, expected, true);
  DelayInternal<TAgent>(mode, expected, false);
}

#endif  // COPY_DELAY_H_
