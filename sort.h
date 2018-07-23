#ifndef SORT_H_
#define SORT_H_

#include <algorithm>
#include <iterator>
#include <parallel/algorithm>

#include "common.h"

// -----------------------------------------------------------------------------
void Sort() {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  FlushCache();

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(agents.begin(), agents.end(), g);
  Timer timer("sort    ");
  // https://gcc.gnu.org/onlinedocs/libstdc++/manual/parallel_mode_using.html#parallel_mode.using.specific
  __gnu_parallel::sort(agents.begin(), agents.end());
}

// -----------------------------------------------------------------------------
void SortMinCopies() {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);
  FlushCache();

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(agents.begin(), agents.end(), g);

  decltype(agents) sorted;
  sorted.resize(agents.size());

  std::vector<uint32_t> uuids;
  uuids.reserve(agents.size());
  for (uint64_t i = 0; i < agents.size(); i++) {
    uuids.push_back(agents[i].GetUuid() % agents.size());
  }

  Timer timer("sort MC ");
  // https://gcc.gnu.org/onlinedocs/libstdc++/manual/parallel_mode_using.html#parallel_mode.using.specific
  __gnu_parallel::sort(uuids.begin(), uuids.end());

#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    sorted[i] = agents[uuids[i]];
  }
}

#endif  // SORT_H_
