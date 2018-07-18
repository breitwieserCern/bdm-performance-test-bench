#ifndef TIMING_H_
#define TIMING_H_

#include <chrono>
#include <iostream>
#include <string>

class Timer {
 public:
  typedef std::chrono::high_resolution_clock Clock;

  explicit Timer(const std::string& description = "")
      : start_{Timestamp()}, text_{description} {}

  ~Timer() {
    int64_t duration = (Timestamp() - start_);
    std::cout << text_ << " " << duration << " ms" << std::endl;
  }

  int64_t Timestamp() {
    using std::chrono::milliseconds;
    using std::chrono::duration_cast;
    auto time = Clock::now();
    auto since_epoch = time.time_since_epoch();
    auto millis = duration_cast<milliseconds>(since_epoch);
    return millis.count();
  }

 private:
  int64_t start_;
  std::string text_;
};

#endif  // TIMING_H_
