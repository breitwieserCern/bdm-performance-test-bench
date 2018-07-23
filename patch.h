#ifndef PATCH_H_
#define PATCH_H_

#include "common.h"
#include "timer.h"

template <typename TWorkload>
void Patch(NeighborMode mode, TWorkload workload, uint64_t reuse,
           double expected) {
  const uint64_t num_agents = Param::num_agents_;

  auto for_each_neighbor = [](uint64_t current_idx, std::vector<Agent>* patch,
                              auto workload_per_agent) {
    double sum = 0;
    for (uint64_t i = 1; i < patch->size(); i++) {
      sum += workload_per_agent(&((*patch)[i]));
    }
    return sum;
  };

  auto add_neighbors_to_patch = [&mode](const auto& agents, auto* patch,
                                        uint64_t current_idx) {
    for (uint64_t i = 1; i < Param::neighbors_per_agent_ + 1; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      (*patch)[i] = agents[nidx];
    }
  };

  auto write_back_patch = [&mode](auto* agents, const auto& patch,
                                  uint64_t current_idx) {
    (*agents)[current_idx] = patch[0];
    for (uint64_t i = 1; i < Param::neighbors_per_agent_; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      (*agents)[nidx] = patch[i];
    }
  };

  std::vector<Agent> agents = Agent::Create(num_agents);
  FlushCache();

  thread_local std::vector<Agent> patch;
  thread_local std::vector<Agent> copy;
  thread_local std::vector<Agent> write_back_cache;
  thread_local double tl_sum = 0;

#pragma omp parallel
  {
    patch.resize(Param::neighbors_per_agent_ + 1);
    write_back_cache.clear();
    tl_sum = 0;
  }
  std::string padding = reuse < 10 ? " " : "";
  Timer timer(padding + std::to_string(reuse) + " Patch");
#pragma omp parallel for
  for (uint64_t i = 0; i < num_agents; i += (reuse + 1)) {
    patch[0] = agents[i];
    add_neighbors_to_patch(agents, &patch, i);

    if (reuse == 0) {
      tl_sum += workload(for_each_neighbor, &patch, 0);
      write_back_patch(&agents, patch, i);
    } else {
      copy = patch;
      for (uint64_t r = 0; r < reuse + 1 && r + i < num_agents; r++) {
        tl_sum += workload(for_each_neighbor, &copy, 0);
        write_back_cache = copy;
        // for (uint64_t el = 0; el < Param::neighbors_per_agent_; el++) {
        //   write_back_cache[el] += copy[el];
        // }
      }
      write_back_patch(&agents, write_back_cache, i);
    }
  }

  double total_sum = 0;
#pragma omp parallel
  {
#pragma omp critical
    total_sum += tl_sum;
  }
  EXPECT_NEAR(total_sum, expected);
}

#endif  // PATCH_H_
