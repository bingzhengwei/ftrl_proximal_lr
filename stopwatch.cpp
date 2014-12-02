#include "stopwatch.h"

double StopWatch::ToSeconds() {
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_).count();
	return ((double)duration /(double)1000000.0);
}

double StopWatch::ToMicroSeconds() {
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_ - start_).count();
	return ((double)duration);
}

StopWatch::StopWatch() {
	StartTimer();
}

void StopWatch::StartTimer() {
	start_ = std::chrono::high_resolution_clock::now();
}

double StopWatch::StopTimer() {
	stop_ = std::chrono::high_resolution_clock::now();
	return ToSeconds();
}

double StopWatch::ElapsedTime() {
	return ToSeconds() ;
}

double StopWatch::ElapsedTimeMS() {
	return ToMicroSeconds() ;
}
