#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <chrono>

class StopWatch {
protected:
	std::chrono::high_resolution_clock::time_point start_;
	std::chrono::high_resolution_clock::time_point stop_;

protected:
	double ToSeconds();
	double ToMicroSeconds();

public:
	StopWatch();

	void StartTimer();
	double StopTimer();
	double ElapsedTime();
	double ElapsedTimeMS();
};

#endif // __STOPWATCH_H__
