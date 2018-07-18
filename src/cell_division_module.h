// -----------------------------------------------------------------------------
//
// Copyright (C) The BioDynaMo Project.
// All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// See the LICENSE file distributed with this work for details.
// See the NOTICE file distributed with this work for additional information
// regarding copyright ownership.
//
// -----------------------------------------------------------------------------

#ifndef DEMO_CELL_DIVISION_MODULE_H_
#define DEMO_CELL_DIVISION_MODULE_H_

#include "biodynamo.h"

namespace bdm {

// -----------------------------------------------------------------------------
class Agent {
 public:
  Agent() {
    for(uint64_t i = 0; i < 12; i++) {
       data_[i] = 1;
    }
    for(uint64_t i = 0; i < 24; i++) {
       data_1_[i] = i;
    }
  }

  double Compute() {
    double sum = 0;
    for (int i = 0; i < 12; i++) {
      sum += data_[i];
    }
    return sum / 12.0;
  }

 private:
  double data_[12];
  uint64_t data_1_[24];
};

inline void FlushCache() {
  const uint64_t bigger_than_cachesize = 100 * 1024 * 1024;
  char *buffer = new char[bigger_than_cachesize];
  for(uint64_t i = 0; i < bigger_than_cachesize; i++) {
     buffer[i] = rand();
  }
  delete buffer;
}

enum NeighborMode { kConsecutive, kRandom };

// inline void CreateNeighbors(NeighborMode mode, uint64_t num_agents, uint64_t neighbors_per_agent, std::vector<uint64_t>* neighbor_indices) {
//   neighbor_indices->resize(num_agents * neighbors_per_agent);
//   switch(mode) {
//     case kConsecutive:
//     for(uint64_t i = 0; i < num_agents; i++) {
//       for(uint64_t j = 0; j < neighbors_per_agent; j++) {
//         (*neighbor_indices)[i * neighbors_per_agent + j] = i + j;
//       }
//     }
//     break;
//
//     case kRandom:
//     Random random;
//     for(uint64_t i = 0; i < neighbor_indices->size(); i++) {
//         (*neighbor_indices)[i] = std::floor(random.Uniform(num_agents));
//     }
//     break;
//   }
// }

inline uint64_t NeighborIndex(NeighborMode mode, uint64_t num_agents, uint64_t current_idx, uint64_t num_neighbor) {
  if(mode == kConsecutive) {
    return std::min(num_agents, current_idx + num_neighbor + 1);
  } else if (mode == kRandom) {
     static Random random;
     return static_cast <uint64_t> (std::floor(random.Uniform(num_agents)));
  }
  throw false;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
void Classic(std::vector<Agent>* agents,
             NeighborMode mode,
             uint64_t neighbors_per_agent,
             TWorkload workload) {

  const uint64_t num_agents = agents->size();

  auto for_each_neighbor = [&num_agents, &neighbors_per_agent, &mode](uint64_t current_idx, std::vector<Agent>* agents, auto workload_per_cell) {
    double sum = 0;
    for (uint64_t i = 0; i < neighbors_per_agent; i++) {
      uint64_t nidx = NeighborIndex(mode, num_agents, current_idx, i);
      sum += workload_per_cell(&((*agents)[nidx]));
    }
    return sum;
  };

  Timing timer("classic");
  thread_local double tl_sum = 0;
  #pragma omp parallel for
  for (uint64_t i = 0; i < agents->size(); i++) {
    tl_sum += workload(for_each_neighbor, agents, i);
  }

  double total_sum = 0;
  #pragma omp parallel
  {
    #pragma omp critical
    total_sum += tl_sum;
  }
  std::cout << "    result: " << total_sum << std::endl;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
void Patch(const std::vector<Agent>& agents,
             NeighborMode mode,
             uint64_t neighbors_per_agent,
             TWorkload workload,
             uint64_t reuse) {

  const uint64_t num_agents = agents.size();

  auto for_each_neighbor = [](uint64_t current_idx, std::vector<Agent>* patch, auto workload_per_cell) {
    double sum = 0;
    for (uint64_t i = 1; i < patch->size(); i++) {
      sum += workload_per_cell(&((*patch)[i]));
    }
    return sum;
  };

  auto add_neighbors_to_patch = [&mode](const auto& agents, auto* patch, uint64_t neighbors_per_agent, uint64_t current_idx) {
    uint64_t num_agents = agents.size();
    for(uint64_t i = 0; i < neighbors_per_agent; i++) {
      uint64_t nidx = NeighborIndex(mode, num_agents, current_idx, i);
      patch->push_back(agents[nidx]);
    }
  };

  thread_local std::vector<Agent> patch;
  thread_local std::vector<Agent> copy;
  thread_local double tl_sum = 0;

  #pragma omp parallel
  {
    patch.reserve(neighbors_per_agent);
    tl_sum = 0;
  }

  Timing timer("Patch  ");
  #pragma omp parallel for
  for (uint64_t i = 0; i < num_agents; i += (reuse + 1)) {
    patch.clear();
    patch.push_back(agents[i]);
    add_neighbors_to_patch(agents, &patch, neighbors_per_agent, i);

    if(reuse == 0) {
      tl_sum += workload(for_each_neighbor, &patch, 0);
    } else {
      copy = patch;
      for (uint64_t r = 0; r < reuse + 1 && r + i < num_agents; r++) {
        tl_sum += workload(for_each_neighbor, &patch, 0);
      }
    }
  }

  double total_sum = 0;
  #pragma omp parallel
  {
    #pragma omp critical
    total_sum += tl_sum;
  }
  std::cout << "    result: " << total_sum << std::endl;
  std::cout << "    reuse : " << reuse << std::endl;
}

// -----------------------------------------------------------------------------
template <typename TWorkload>
inline void Run(uint64_t num_agents, uint64_t neighbors_per_agent, NeighborMode mode, TWorkload workload) {
  std::vector<Agent> agents;
  agents.resize(num_agents);

  std::vector<uint64_t> neighbor_indices;

  FlushCache();

  Classic(&agents, mode, neighbors_per_agent, workload);

  std::vector<uint64_t> reuse_vals = {0, 1, 2, 4, 8, 16, 64};
  for(auto& r : reuse_vals) {
    FlushCache();
    Patch(agents, mode, neighbors_per_agent, workload, r);
  }
}

// -----------------------------------------------------------------------------
inline int Simulate(int argc, const char** argv) {
  uint64_t num_agents =  1e7;
  uint64_t neighbors_per_agent = 9;

  if (argc == 3) {
    num_agents = std::atoi(argv[1]);
    neighbors_per_agent = std::atoi(argv[2]);
  }

  auto workload_per_cell = [](Agent* current) {
    double sum = 0;
    sum += current->Compute();
    return sum;
  };

  auto workload = [&](auto for_each_neighbor, std::vector<Agent>* agents, uint64_t current_idx) {
    double sum = 0;
    Agent* current = &((*agents)[current_idx]);
    sum += workload_per_cell(current);
    sum += for_each_neighbor(current_idx, agents, workload_per_cell);
    return sum;
  };

  Agent a;
  std::cout << "Result for one agent: " << workload_per_cell(&a) << std::endl;

  // Run(num_agents, neighbors_per_agent, kRandom, workload);
  Run(num_agents, neighbors_per_agent, kConsecutive, workload);

  return 0;
}

// // -----------------------------------------------------------------------------
// template <typename T, uint64_t N>
// class MyVector {
// public:
//   MyVector() {}
//   MyVector& operator=(const MyVector& other) {
//     for(uint64_t i = 0; i < other.size_; i++) {
//       data_[i] = other.data_[i];
//     }
//   }
//   void push_back(const T& element) {
//     data_[size_++] = element;
//   }
//   uint64_t size() const { return size_; }
//   void clear() { size_ = 0; }
//   void reserve(int) {}
//  private:
//   uint64_t size_ = 0;
//   T data_[N];
// };

// // -----------------------------------------------------------------------------
// inline void Run(int c) {
//   auto* simulation = Simulation<>::GetActive();
//   auto* rm = simulation->GetResourceManager();
//   auto* grid= simulation->GetGrid();
//   grid->Initialize();
//   auto* cells = rm->Get<Cell>();
//
//   std::vector<Foo> last;
//   const uint64_t N = 256 * 256 * 256;
//   last.resize(N);
//   // last.resize(cells->size());
//
//   // maps uuid to index in array
//   // indirection increased runtime from 1350 to 1900
//   // std::unordered_map<uint64_t, uint64_t> m;
//   // for (uint64_t i = 0; i < cells->size(); i++) {
//   //   m[i] = i;
//   // }
//
//   // std::cout << "size of Foo: " << sizeof(Foo) << std::endl;
//   std::cout << "c " << c << std::endl;
//   // thread_local MyVector<Foo, 65> patch;
//   thread_local std::vector<Foo> patch;
//   thread_local std::vector<decltype(patch)> copy;
//
//   #pragma omp parallel
//   {
//     patch.reserve(64);
//     copy.resize(c);
//   }
//
//   auto add_to_patch = [&](SoHandle neighbor_handle) {
//     patch.push_back(last[neighbor_handle.GetElementIdx()]);
//     // rm->ApplyOnElement(neighbor_handle, [&](auto&& neighbor) {
//     //   patch.emplace_back(neighbor);
//     // });
//   };
//
//   auto add_consecutive = [&](uint64_t idx, uint64_t num_elements) {
//     for(uint64_t i = idx; i < idx + num_elements; i++) {
//       patch.push_back(last[i]);
//     }
//   };
//
//   int counter = 0;
//   auto num_neighbors = [&](SoHandle neighbor_handle) {
//     counter++;
//   };
//
//   Timing t("   runtime ");
//
//   // #pragma omp parallel for
//   // for(uint64_t i = 0; i < last.size() - 64; i += (c + 1)) {
//   for(uint64_t i = 0; i < cells->size(); i++) {
//     // patch.clear();
//     //
//     // patch.push_back(last[i]);
//     // add_consecutive(i, 64);
//     //
//     // for (uint64_t j = 0; j < copy.size(); j++) {
//     //   copy[j] = patch;
//     // }
//
//     // patch.emplace_back(cell);
//     auto&& cell = (*cells)[i];
//     // grid->ForEachNeighbor(add_to_patch, cell, SoHandle(0, i));
//
//     grid->ForEachNeighbor(num_neighbors, cell, SoHandle(0, i));
//   }
//   std::cout << counter << std::endl;
//   std::cout << "avg num neighbors " << (counter / cells->size()) << std::endl;
//   // std::cout << "patch size " <<  patch.size() << std::endl;
// }
//
//
// // -----------------------------------------------------------------------------
// inline int Simulate(int argc, const char** argv) {
//   // 2. Create new simulation
//   Simulation<> simulation(argc, argv);
//   simulation.GetParam()->statistics_ = true;
//
//   // 3. Define initial model - in this example: 3D grid of cells
//   size_t cells_per_dim = 256;
//   auto construct = [](const std::array<double, 3>& position) {
//     Cell cell(position);
//     cell.SetDiameter(30);
//     cell.SetAdherence(0.4);
//     cell.SetMass(1.0);
//     // cell.AddBiologyModule(GrowDivide());
//     return cell;
//   };
//   {
//     Timing t("init");
//     ModelInitializer::Grid3D(cells_per_dim, 20, construct);
//   }
//
//   // 4. Run simulation for one timestep
//   simulation.GetScheduler()->Simulate(1);
//
//   Run(0);
//   // for(int i = 1; i <= 64; i *= 2) {
//   //   Run(i);
//   // }
//
//   std::cout << "Simulation completed successfully!\n";
//   return 0;
// }

}  // namespace bdm

#endif  // DEMO_CELL_DIVISION_MODULE_H_
