#ifndef __UTIL_H__
#define __UTIL_H__

#ifndef __APPLE__
#include <malloc.h>
#endif
#include <memory.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <emmintrin.h>
#endif

#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <future>
#include <algorithm>

#define SSE_ALIGN_SIZE 16
#define MAX_EXP_NUM 50.

template<class Func>
void util_parallel_for(size_t start, size_t end, const Func& func) {
	auto thread_num = std::thread::hardware_concurrency();
	std::vector<std::future<void>> futures;
	size_t thread_batch_size = (end - start) / thread_num;

	for (size_t i = 0; i < thread_num; ++i) {
		size_t s = start + i * thread_batch_size;
		size_t e = start + (i + 1) * thread_batch_size;
		if (i == thread_num - 1) {
			e = end;
		}

		futures.emplace_back( std::async([&, s, e]() {
			for (size_t j = s; j < e; ++j) {
			func(j);
			}
		}));
	}

	for (auto&& f : futures) {
		f.wait();
	}
}

template<class Func>
void util_for(size_t start, size_t end, const Func& func) {
	for(size_t i = start; i < end; ++i) {
		func(i);
	}
}

inline static void * aligned_malloc_internal(size_t size) {
#if defined(_MSC_VER)
	void *p = _aligned_malloc(size, SSE_ALIGN_SIZE);
#elif defined(__APPLE__)  /* OS X always aligns on 16-byte boundaries */
	void *p = malloc(size);
#else
	void *p = NULL, *ptr = NULL;
	if (posix_memalign(&ptr, SSE_ALIGN_SIZE, size) == 0) {
		p = ptr;
	}
#endif
	return p;
}

inline static void aligned_free_internal(void * p) {
#ifdef _MSC_VER
	_aligned_free(p);
#else
	free(p);
#endif
}

inline size_t round_up(size_t n) {
	n += 7;
	n /= 8;
	n *= 8;
	return n;
}

inline double* sse_malloc(size_t n) {
	return (double *) aligned_malloc_internal(sizeof(double) * round_up(n));
}

inline void sse_free(double *p) {
	if (p) {
		aligned_free_internal(p);
	}
}

inline static void vector_set(double *x, const double c, size_t n) {
	__m128d mc = _mm_set1_pd(c);
	for (size_t i = 0; i < n; i += 8) {
		_mm_store_pd(x + i, mc);
		_mm_store_pd(x + i + 2, mc);
		_mm_store_pd(x + i + 4, mc);
		_mm_store_pd(x + i + 6, mc);
	}
}

//y[i] = x[i]
inline static void vector_copy(double *y, double *x, size_t n) {
	for (size_t i = 0; i < n; i += 8) {
		__m128d x0 = _mm_load_pd(x + i);
		__m128d x1 = _mm_load_pd(x + i + 2);
		__m128d x2 = _mm_load_pd(x + i + 4);
		__m128d x3 = _mm_load_pd(x + i + 6);
		_mm_store_pd(y + i, x0);
		_mm_store_pd(y + i + 2, x1);
		_mm_store_pd(y + i + 4, x2);
		_mm_store_pd(y + i + 6, x3);
	}
}

inline bool double_equal(const double d1, const double d2) {
	return std::fabs(d1 - d2) < std::numeric_limits<double>::epsilon();
}

inline bool double_greater(const double d1, const double d2) {
	if (double_equal(d1, d2)) {
		return false;
	}

	return d1 > d2;
}

inline int double_cmp(const double d1, const double d2) {
	if (double_equal(d1, d2)) {
		return 0;
	}
	if (d1 > d2) {
		return 1;
	}
	return -1;
}

inline bool double_greater_equal(const double d1, const double d2) {
	if (double_equal(d1, d2)) {
		return true;
	}

	return d1 > d2;
}

inline bool double_less(const double d1, const double d2) {
	if (double_equal(d1, d2)) {
		return false;
	}

	return d1 < d2;
}

inline bool double_less_equal(const double d1, const double d2) {
	if (double_equal(d1, d2)) {
		return true;
	}

	return d1 < d2;
}

inline double safe_exp(double x) {
	return std::exp(std::max(std::min(x, MAX_EXP_NUM), -MAX_EXP_NUM));
}

#endif // __UTIL_H__
