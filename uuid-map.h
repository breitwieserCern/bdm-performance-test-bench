#ifndef BUILD_UUID_MAP_H_
#define BUILD_UUID_MAP_H_

#include <unordered_map>
#include "common.h"

struct Location {
  Location() : container_idx_(0), element_idx_(0) {}
  Location(uint64_t element_idx)
      : container_idx_{0}, element_idx_{element_idx} {}
  uint16_t container_idx_;
  uint64_t element_idx_;
  friend std::ostream& operator<<(std::ostream& str, const Location l) {
    str << "{ cidx: " << l.container_idx_ << ", eidx: " << l.element_idx_
        << "}";
    return str;
  }
};

void UuidMap() {
  auto&& agents = SoaAgent::Create(Param::num_agents_);
  FlushCache();

  std::cout << "uuid map\n";

  // uuid -> Location
  std::unordered_map<uint64_t, Location> map;
  {
    FlushCache();
    Timer timer("  build map   ");
    map.reserve(agents.size() * 1.5);

    for (uint64_t i = 0; i < agents.size(); i++) {
      map[agents[i].GetUuid()] = i;
    }
  }

  decltype(map) cpy_map;
  {
    FlushCache();
    Timer timer("  cpy map     ");
    cpy_map = map;
  }

  {
    FlushCache();
    Timer timer("  cpy again   ");
    cpy_map[123] = 321;
    cpy_map = map;
  }

  const uint64_t elements_per_halo = 1e5;
  uint64_t cnt = agents.size();
  std::array<decltype(map), 26> halo_regions;
  {
    FlushCache();
    Timer timer("  build halos ");
    for (auto& halo : halo_regions) {
      halo.reserve(elements_per_halo);
      for (uint64_t i = 0; i < elements_per_halo; i++) {
        halo[i + cnt] = i;
      }
      cnt += elements_per_halo;
    }
  }

  {
    FlushCache();
    Timer timer("  iterate     ");
    double sum = 0;
    for (uint64_t i = 0; i < cnt; i++) {
      const auto& search = map.find(i);
      if (search != map.end()) {
        sum += search->second.element_idx_;
        continue;
      }
      for (auto&& halo : halo_regions) {
        const auto& search = halo.find(i);
        if (search != halo.end()) {
          sum += search->second.element_idx_;
          break;
        }
      }
    }
    std::cout << "    " << sum << "\n";
  }

  decltype(map) merged = std::move(map);
  {
    FlushCache();
    Timer timer("  merge maps  ");
    for (auto&& halo : halo_regions) {
      merged.merge(halo);
    }
  }

  {
    FlushCache();
    Timer timer("  iterate2    ");
    double sum = 0;
    for (uint64_t i = 0; i < cnt; i++) {
      sum += merged[i].element_idx_;
    }
    std::cout << "    " << sum << "\n";
  }

  // just for comparison
  {
    FlushCache();
    Timer timer("  new vector  ");
    std::vector<uint64_t> v;
    v.reserve(agents.size());
    for (uint64_t i = 0; i < agents.size(); i++) {
      v.push_back(i);
    }
  }

  // some printouts to avoid that the compiler removes whole sections because
  // they are not used
  std::cout << map[0] << "\n";
  std::cout << cpy_map[10] << "\n";
}

#endif  // BUILD_UUID_MAP_H_
