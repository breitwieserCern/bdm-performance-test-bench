#ifndef PATCH_H_
#define PATCH_H_

#include "common.h"
#include "timer.h"

template <typename TAgent, typename TWorkload>
void Patch(NeighborMode mode, TWorkload workload, uint64_t reuse,
           double expected) {
  const uint64_t num_agents = Param::num_agents_;
  auto&& agents = TAgent::Create(num_agents);
  auto&& agents_t1 = TAgent::Create(num_agents);

  auto for_each_neighbor = [](uint64_t current_idx, auto* patch) {
    double sum = 0;
    for (uint64_t i = 1; i < Param::mutated_neighbors_ + 1; i++) {
      sum += (*patch)[i].ComputeNeighbor();
    }
    for (uint64_t i = Param::mutated_neighbors_ + 1; i < patch->size(); i++) {
      sum += (*patch)[i].ComputeNeighborReadPart();
    }
    return sum;
  };

  auto add_neighbors_to_patch = [&mode](auto& agents, auto* patch,
                                        uint64_t current_idx) {
    for (uint64_t i = 1; i < Param::neighbors_per_agent_ + 1; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i);
      (*patch)[i] = agents[nidx];
    }
  };

  auto write_back_patch = [&mode, &agents](auto* dest, auto& patch,
                                           uint64_t current_idx) {
    (*dest)[current_idx].ApplyDelta(agents[current_idx], patch[0]);
    for (uint64_t i = 1; i < Param::mutated_neighbors_ + 1; i++) {
      uint64_t nidx = NeighborIndex(mode, current_idx, i - 1);
      (*dest)[nidx].ApplyDelta(agents[nidx], patch[i]);
    }
  };

  FlushCache();

  using AgentContainer = std::decay_t<decltype(agents)>;

  thread_local AgentContainer patch;
  thread_local AgentContainer copy;
  thread_local AgentContainer write_back_cache;
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
      write_back_patch(&agents_t1, patch, i);
    } else {
      copy = patch;
      for (uint64_t r = 0; r < reuse + 1 && r + i < num_agents; r++) {
        patch = copy;
        tl_sum += workload(for_each_neighbor, &patch, 0);
        if (r == 0) {
          write_back_cache = patch;
        } else {
          for (uint64_t el = 0; el < Param::mutated_neighbors_ + 1; el++) {
            write_back_cache[el].ApplyDelta(copy[el], patch[el]);
          }
        }
      }
      write_back_patch(&agents_t1, write_back_cache, i);
    }
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

#endif  // PATCH_H_
