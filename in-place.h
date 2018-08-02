#ifndef IN_PLACE_H_
#define IN_PLACE_H_

#include "common.h"
#include "timer.h"

/// This is the current solution that does not take discretization into account
/// and will lead to race conditions if neighbors are modified. This is the
/// baseline to quantify solutions that solve the two issues.
/// \tparam TAgent either Agent or SoaAgent
template <typename TAgent, typename TWorkload>
void InPlace(NeighborMode mode, TWorkload workload, double expected) {
  auto for_each_neighbor = [&mode](uint64_t current_idx, auto* agents) {
    double sum = 0;
    for (uint64_t i = 0; i < Param::mutated_neighbors_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      sum += (*agents)[nidx].ComputeNeighbor();
    }
    for (uint64_t i = Param::mutated_neighbors_;
         i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      sum += (*agents)[nidx].ComputeNeighborReadPart();
    }
    return sum;
  };

  auto&& agents = TAgent::Create(Param::num_agents_);
  FlushCache();

  thread_local double tl_sum = 0;
#pragma omp parallel
  tl_sum = 0;

  Timer timer("inplace ", true);
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

  // check data member values
  double checksum = 0;
  for (uint64_t i = 0; i < agents.size(); i++) {
    checksum += agents[i].CheckSum();
  }
  total_sum += checksum;

  EXPECT_NEAR(total_sum, expected);
}

#endif  // IN_PLACE_H_
