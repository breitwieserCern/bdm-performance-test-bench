#ifndef TIMING_H_
#define TIMING_H_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

class Timer {
 public:
  typedef std::chrono::high_resolution_clock Clock;

  explicit Timer(const std::string& description = "",
                 bool set_reference = false)
      : start_{Timestamp()},
        set_reference_(set_reference),
        text_{description} {}

  ~Timer() {
    int64_t duration = (Timestamp() - start_);
    std::cout << text_ << " " << std::setfill(' ') << std::setw(5) << duration
              << " ms";
    if (reference_ != 0) {
      std::cout << "  - slowdown " << std::setprecision(2)
                << duration / reference_;
    }
    std::cout << std::endl;
    if (set_reference_) {
      reference_ = duration;
    }
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
  bool set_reference_ = false;
  static double reference_;
  std::string text_;
};

double Timer::reference_ = 0;

#endif  // TIMING_H_
