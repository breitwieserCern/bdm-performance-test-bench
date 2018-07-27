#ifndef EXTRACT_DATAMEMBER_H_
#define EXTRACT_DATAMEMBER_H_

void ExtractDatamember() {
  std::vector<Agent> agents = Agent::Create(Param::num_agents_);

  Timer timer("extract ");
  std::vector<double> data;
  data.resize(Param::num_agents_);

#pragma omp parallel for
  for (uint64_t i = 0; i < agents.size(); i++) {
    data[i] = agents[i].GetElement();
  }

  std::cout << data[10] << std::endl;
}

#endif  // EXTRACT_DATAMEMBER_H_
