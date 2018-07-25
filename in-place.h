#ifndef IN_PLACE_H_
#define IN_PLACE_H_

#include "common.h"
#include "timer.h"

template <typename TWorkload>
void InPlace(NeighborMode mode, TWorkload workload, double expected) {
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

  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
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

  EXPECT_NEAR(total_sum, expected);
}

#endif  // IN_PLACE_H_
