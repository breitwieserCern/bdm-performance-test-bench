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
// This model creates a grid of 128x128x128 cells. Each cell grows untill a
// specific volume, after which it proliferates (i.e. divides).
// -----------------------------------------------------------------------------

// 1. Define compile time parameter
template <typename Backend>
struct CompileTimeParam : public DefaultCompileTimeParam<Backend> {
  // use predefined biology module GrowDivide
  using BiologyModules = Variant<GrowDivide>;
  using SimulationBackend = Scalar;
  // use default Backend and AtomicTypes
};

struct Foo {
  double data_[12];
  uint64_t data_1[24];
};

inline void Run(int c) {
  auto* simulation = Simulation<>::GetActive();
  auto* rm = simulation->GetResourceManager();
  auto* grid= simulation->GetGrid();
  grid->Initialize();


  auto* cells = rm->Get<Cell>();
  std::vector<Foo> last;
  last.resize(cells->size());

  // maps uuid to index in array
  // indirection increased runtime from 1350 to 1900
  std::unordered_map<uint64_t, uint64_t> m;
  for (uint64_t i = 0; i < cells->size(); i++) {
    m[i] = i;
  }

  // std::cout << "size of Foo: " << sizeof(Foo) << std::endl;
  std::cout << "c " << c << std::endl;
  Timing t("copy");
  thread_local std::vector<Foo> patch;
  thread_local std::vector<decltype(patch)> copy;

  #pragma omp parallel
  {
    patch.reserve(64);
    copy.resize(c);
  }

  auto add_to_patch = [&](SoHandle neighbor_handle) {
    patch.emplace_back(last[m[neighbor_handle.GetElementIdx()]]);
    // rm->ApplyOnElement(neighbor_handle, [&](auto&& neighbor) {
    //   patch.emplace_back(neighbor);
    // });
  };

  auto add_consecutive = [&m](const std::vector<Foo>& last, std::vector<Foo>* patch,  uint64_t idx, uint64_t num_elements) {
    for(uint64_t i = idx; i < idx + num_elements; i++) {
      patch->emplace_back(last[m[i]]);
    }
  };

  #pragma omp parallel for
  for(uint64_t i = 0; i < cells->size() - 64; i += (copy.size() + 1)) {
    patch.clear();


    patch.emplace_back(last[m[i]]);
    add_consecutive(last, &patch, i, 64);

    for (uint64_t j = 0; j < copy.size(); j++) {
      copy[j] = patch;
    }

    // patch.emplace_back(cell);
    // auto&& cell = (*cells)[i];
    // grid->ForEachNeighbor(add_to_patch, cell, SoHandle(0, i));
  }
  std::cout << patch.size() << std::endl;
}

inline int Simulate(int argc, const char** argv) {
  // 2. Create new simulation
  Simulation<> simulation(argc, argv);
  simulation.GetParam()->statistics_ = true;

  // 3. Define initial model - in this example: 3D grid of cells
  size_t cells_per_dim = 128;
  auto construct = [](const std::array<double, 3>& position) {
    Cell cell(position);
    cell.SetDiameter(30);
    cell.SetAdherence(0.4);
    cell.SetMass(1.0);
    cell.AddBiologyModule(GrowDivide());
    return cell;
  };
  {
    Timing t("init");
    ModelInitializer::Grid3D(cells_per_dim, 20, construct);
  }


  // 4. Run simulation for one timestep
  simulation.GetScheduler()->Simulate(1);

  for(int i = 0; i < 20; i+=5) {
    Run(i);
  }

  std::cout << "Simulation completed successfully!\n";
  return 0;
}

}  // namespace bdm

#endif  // DEMO_CELL_DIVISION_MODULE_H_
