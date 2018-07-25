#ifndef COPY_H_
#define COPY_H_

#include "common.h"
#include "timer.h"

void Copy(NeighborMode mode, double expected) {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  std::vector<Agent> agents_t1 = Agent::Create(Param::num_agents_);

  auto for_each_neighbor = [&mode, &agents_t1](uint64_t current_idx,
                                               std::vector<Agent>* agents) {
    double sum = 0;
    for (uint64_t i = 0; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      Agent neighbor_cpy = (*agents)[nidx];
      sum += neighbor_cpy.ComputeNeighbor();
      agents_t1[nidx] = neighbor_cpy;
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
    agents_t1[i] = copy;
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }

  EXPECT_NEAR(total_sum, expected);
}

#endif  // COPY_H_