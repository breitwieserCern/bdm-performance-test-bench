#ifndef PARAM_H_
#define PARAM_H_

struct Param {
  static uint64_t num_agents_;
  static uint64_t neighbors_per_agent_;
  /// number of neighbors that will be mutated. must be smaller than
  /// neighbors_per_agent_
  static uint64_t mutated_neighbors_;
  static uint64_t num_neighbor_ops_;
  static uint64_t neighbor_range_;
  static bool compute_intense_;
};

uint64_t Param::num_agents_ = 1e7;
uint64_t Param::neighbors_per_agent_ = 64;
uint64_t Param::mutated_neighbors_ = 64;
uint64_t Param::num_neighbor_ops_ = 1;
uint64_t Param::neighbor_range_ = 1e3;
bool Param::compute_intense_ = false;

#endif  // PARAM_H_
