#ifndef TWO_PASSES_H_
#define TWO_PASSES_H_

#include <functional>
#include <unordered_map>

#include "common.h"
#include "timer.h"

/// Delay modifying calls to the next itereration. Stores the call information
/// in the afector not in the afectee. Therefore, neighbors are not modified.
/// In the next iteration each cell asks each neighbor if it has delayed calls
/// for itself.
/// Assumes that calling non-const functions on neighbors can be delayed.
/// Discretization issue is solved by copying the current agent.
/// There are no locks needed.
/// Uses std::function to capture parameters of the delayed function calls.
/// NB: This simplification doesn't pass the result check because the neighbor
///     relations are not symmetric (Neighbor(A): B != Neighbor(B): A).
///     However, it gives a lower bound runtime expectation.
template <typename TAgent>
void NextIteration(NeighborMode mode, double expected) {
  auto&& agents = TAgent::Create(Param::num_agents_);
  auto&& agents_t1 = TAgent::Create(Param::num_agents_);

  std::vector<std::unordered_map<
      uint64_t,
      std::vector<std::function<void(std::decay_t<decltype(agents)>*)>>>>
      delayed;
  delayed.resize(agents.size());

  auto for_each_neighbor = [&mode, &agents_t1, &delayed](uint64_t current_idx,
                                                         auto* agents) {
    double sum = 0;
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      sum += (*agents)[nidx].ComputeNeighborReadPart();
      if (i < Param::mutated_neighbors_) {
        delayed[current_idx][nidx].push_back(
            [=](auto* agents) { (*agents)[nidx].ComputeNeighborWritePart(); });
      }
    }
    return sum;
  };

  auto workload = [&](auto for_each_neighbor, auto* agents,
                      auto&& current_agent, uint64_t current_idx) {
    double sum = 0;
    sum += current_agent.Compute();
    for (uint64_t i = 0; i < Param::num_neighbor_ops_; i++) {
      sum += for_each_neighbor(current_idx, agents);
    }
    return sum;
  };

  FlushCache();

  thread_local double tl_sum = 0;
  thread_local Agent copy;
#pragma omp parallel
  tl_sum = 0;

  // step 1
  Timer timer("next-it ");
#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    copy = agents[i];
    tl_sum += workload(for_each_neighbor, &agents, copy, i);
    agents_t1[i] = copy;
  }

  // step 2
  auto fen2 = [&mode, &delayed](uint64_t current_idx, auto* agents) {
    double sum = 0;
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      const auto& map = delayed[nidx];
      const auto& search = map.find(current_idx);
      if (search == map.end()) {
        continue;
      }
      for (auto& f : search->second) {
        f(agents);
      }
    }
    return sum;
  };

#pragma omp parallel for
  for (uint64_t i = 0; i < agents_t1.size(); i++) {
    fen2(i, &agents_t1);
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }

  // check data member values
  double checksum = 0;
  for (uint64_t i = 0; i < agents_t1.size(); i++) {
    checksum += agents_t1[i].CheckSum();
  }
  total_sum += checksum;

  EXPECT_NEAR(total_sum, expected);
}

#endif  // TWO_PASSES_H_
