#ifndef PARAM_H_
#define PARAM_H_

struct Param {
  static uint64_t num_agents_;
  static uint64_t neighbors_per_agent_;
  static uint64_t num_neighbor_ops_;
  static uint64_t neighbor_range_;
};

uint64_t Param::num_agents_ = 1e7;
uint64_t Param::neighbors_per_agent_ = 64;
uint64_t Param::num_neighbor_ops_ = 1;
uint64_t Param::neighbor_range_ = 1e3;

#endif  // PARAM_H_
