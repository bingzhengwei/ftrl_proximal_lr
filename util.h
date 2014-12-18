#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <future>
#include <algorithm>

#define MAX_EXP_NUM 50.
#define MIN_SIGMOID (10e-15)
#define MAX_SIGMOID (1. - 10e-15)


template<class Func>
void util_parallel_run(const Func& func, size_t num_threads = 0) {
	if (num_threads == 0) {
		num_threads = std::thread::hardware_concurrency();
	}

	std::thread *threads = new std::thread[num_threads];
	for(size_t i = 0; i < num_threads; ++i) {
		threads[i] = std::thread(func, i);
	}

	for (size_t i = 0; i < num_threads; ++i) {
		threads[i].join();
	}

	delete [] threads;
}

template<typename T>
inline bool util_equal(const T v1, const T v2) {
	return std::fabs(v1 - v2) < std::numeric_limits<T>::epsilon();
}

template<typename T>
inline bool util_greater(const T v1, const T v2) {
	if (util_equal(v1, v2)) {
		return false;
	}

	return v1 > v2;
}

template<typename T>
inline int util_cmp(const T v1, const T v2) {
	if (util_equal(v1, v2)) {
		return 0;
	} else if (v1 > v2) {
		return 1;
	} else {
		return -1;
	}
}

template<typename T>
inline bool util_greater_equal(const T v1, const T v2) {
	if (util_equal(v1, v2)) {
		return true;
	}

	return v1 > v2;
}

template<typename T>
inline bool util_less(const T v1, const T v2) {
	if (util_equal(v1, v2)) {
		return false;
	}

	return v1 < v2;
}

template<typename T>
inline bool util_less_equal(const T v1, const T v2) {
	if (util_equal(v1, v2)) {
		return true;
	}

	return v1 < v2;
}

template<typename T>
inline T safe_exp(T x) {
	return std::exp(std::max(std::min(x, (T)MAX_EXP_NUM), -(T)MAX_EXP_NUM));
}

template<typename T>
inline T sigmoid(T x) {
	return (T)1. / ((T)1. + safe_exp(-x));
}

#endif // __UTIL_H__
