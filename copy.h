#ifndef COPY_H_
#define COPY_H_

#include "common.h"
#include "timer.h"

template <typename TAgent>
void Copy(NeighborMode mode, double expected) {
  auto&& agents = TAgent::Create(Param::num_agents_);
  auto&& agents_t1 = TAgent::Create(Param::num_agents_);

  auto for_each_neighbor = [&mode, &agents_t1](uint64_t current_idx,
                                               auto* agents) {
    double sum = 0;
    for (uint64_t i = 0; i < Param::mutated_neighbors_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      Agent neighbor_cpy = (*agents)[nidx];
      sum += neighbor_cpy.ComputeNeighbor();
      agents_t1[nidx].ApplyDelta((*agents)[nidx], neighbor_cpy);
    }
    for (uint64_t i = Param::mutated_neighbors_;
         i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      sum += (*agents)[nidx].ComputeNeighborReadPart();
    }
    return sum;
  };

  auto workload = [&](auto for_each_neighbor, auto* agents, auto* current_agent,
                      uint64_t current_idx) {
    double sum = 0;
    sum += current_agent->Compute();
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

  Timer timer("copy    ");
#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    copy = agents[i];
    tl_sum += workload(for_each_neighbor, &agents, &copy, i);
    agents_t1[i].ApplyDelta(agents[i], copy);
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

#endif  // COPY_H_
