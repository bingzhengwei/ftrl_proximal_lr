// Copyright (c) 2014-2015 The AsyncFTRL Project
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_UTIL_H
#define SRC_UTIL_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <future>
#include <limits>
#include <vector>

#define MAX_EXP_NUM 50.
#define MIN_SIGMOID (10e-15)
#define MAX_SIGMOID (1. - 10e-15)


template<class Func>
void util_parallel_run(const Func& func, size_t num_threads = 0) {
	if (num_threads == 0) {
		num_threads = std::thread::hardware_concurrency();
	}

	std::thread *threads = new std::thread[num_threads];
	for (size_t i = 0; i < num_threads; ++i) {
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
	T max_exp = static_cast<T>(MAX_EXP_NUM);
	return std::exp(std::max(std::min(x, max_exp), -max_exp));
}

template<typename T>
inline T sigmoid(T x) {
	T one = 1.;
	return one / (one + safe_exp(-x));
}

#endif // SRC_UTIL_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
