#ifndef _BENCH_HPP_
#define _BENCH_HPP_

#include<numeric>
#include<vector>

namespace benchmark {
  template<typename T>
  struct stats_t
  {
    T total;
    T mean;
    T min;
    T max;
    T stddev;
  };

  template<typename T>
  inline stats_t<T> calculateStatistics(std::vector<T>& dataset,int ntrials)
  {
    stats_t<T> s;
    std::sort(dataset.begin(), dataset.end());
    s.total = std::accumulate(dataset.begin(), dataset.end(),T(0));
    s.mean = s.total / static_cast<T>(dataset.size());
    s.min = dataset.front();
    s.max = dataset.back();

    T var = T(0);
    for(T& x : dataset)
      var += x*x;
    var /= static_cast<T>(dataset.size());
    var -= (s.mean*s.mean);
    s.stddev = std::sqrt(var);
    return s;
  }
}
#endif
