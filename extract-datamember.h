#ifndef EXTRACT_DATAMEMBER_H_
#define EXTRACT_DATAMEMBER_H_

/// Profile the time required to extract one data member within AOS into
/// SOA representation. In case a AOS layout is chosen, this overhead has to be
/// paid potentially to send data to ParaView or the GPU.
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
