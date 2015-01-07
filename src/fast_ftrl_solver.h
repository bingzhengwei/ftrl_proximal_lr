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

#ifndef SRC_FAST_FTRL_SOLVER_H
#define SRC_FAST_FTRL_SOLVER_H

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "src/ftrl_solver.h"
#include "src/lock.h"

enum { kParamGroupSize = 10, kFetchStep = 3, kPushStep = 3 };

inline size_t calc_group_num(size_t n) {
	return (n + kParamGroupSize - 1) / kParamGroupSize;
}

template<typename T>
class FtrlParamServer : public FtrlSolver<T> {
public:
	FtrlParamServer();

	virtual ~FtrlParamServer();

	virtual bool Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout = 0);

	virtual bool Initialize(const char* path);

	bool FetchParamGroup(T* n, T* z, size_t group);

	bool FetchParam(T* n, T* z);

	bool PushParamGroup(T* n, T* z, size_t group);

private:
	size_t param_group_num_;
	SpinLock* lock_slots_;
};

template<typename T>
class FtrlWorker : public FtrlSolver<T> {
public:
	FtrlWorker();

	virtual ~FtrlWorker();

	bool Initialize(
		FtrlParamServer<T>* param_server,
		size_t push_step = kPushStep,
		size_t fetch_step = kFetchStep);

	bool Reset(FtrlParamServer<T>* param_server);

	bool Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout = 0) { return false; }

	bool Initialize(const char* path) { return false; }

	T Update(const std::vector<std::pair<size_t, T> >& x, T y) { return false; }

	T Update(
		const std::vector<std::pair<size_t, T> >& x,
		T y,
		FtrlParamServer<T>* param_server);

	bool PushParam(FtrlParamServer<T>* param_server);

private:
	size_t param_group_num_;
	size_t* param_group_step_;
	size_t push_step_;
	size_t fetch_step_;

	T * n_update_;
	T * z_update_;
};



template<typename T>
FtrlParamServer<T>::FtrlParamServer()
: FtrlSolver<T>(), param_group_num_(0), lock_slots_(NULL) {}

template<typename T>
FtrlParamServer<T>::~FtrlParamServer() {
	if (lock_slots_) {
		delete [] lock_slots_;
	}
}

template<typename T>
bool FtrlParamServer<T>::Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout) {
	if(!FtrlSolver<T>::Initialize(alpha, beta, l1, l2, n, dropout)) {
		return false;
	}

	param_group_num_ = calc_group_num(n);
	lock_slots_ = new SpinLock[param_group_num_];

	FtrlSolver<T>::init_ = true;
	return true;
}

template<typename T>
bool FtrlParamServer<T>::Initialize(const char* path) {
	if(!FtrlSolver<T>::Initialize(path)) {
		return false;
	}

	param_group_num_ = calc_group_num(FtrlSolver<T>::feat_num_);
	lock_slots_ = new SpinLock[param_group_num_];

	FtrlSolver<T>::init_ = true;
	return true;
}

template<typename T>
bool FtrlParamServer<T>::FetchParamGroup(T* n, T* z, size_t group) {
	if (!FtrlSolver<T>::init_) return false;

	size_t start = group * kParamGroupSize;
	size_t end = std::min((group + 1) * kParamGroupSize, FtrlSolver<T>::feat_num_);

	std::lock_guard<SpinLock> lock(lock_slots_[group]);
	for(size_t i = start; i < end; ++i) {
		n[i] = FtrlSolver<T>::n_[i];
		z[i] = FtrlSolver<T>::z_[i];
	}

	return true;
}

template<typename T>
bool FtrlParamServer<T>::FetchParam(T* n, T* z) {
	if (!FtrlSolver<T>::init_) return false;

	for(size_t i = 0; i < param_group_num_; ++i) {
		FetchParamGroup(n, z, i);
	}
	return true;
}

template<typename T>
bool FtrlParamServer<T>::PushParamGroup(T* n, T* z, size_t group) {
	if (!FtrlSolver<T>::init_) return false;

	size_t start = group * kParamGroupSize;
	size_t end = std::min((group + 1) * kParamGroupSize, FtrlSolver<T>::feat_num_);

	std::lock_guard<SpinLock> lock(lock_slots_[group]);
	for(size_t i = start; i < end; ++i) {
		FtrlSolver<T>::n_[i] += n[i];
		FtrlSolver<T>::z_[i] += z[i];
		n[i] = 0;
		z[i] = 0;
	}

	return true;
}


template<typename T>
FtrlWorker<T>::FtrlWorker()
: FtrlSolver<T>(), param_group_num_(0), param_group_step_(NULL),
push_step_(0), fetch_step_(0), n_update_(NULL), z_update_(NULL) {}

template<typename T>
FtrlWorker<T>::~FtrlWorker() {
	if (param_group_step_) {
		delete [] param_group_step_;
	}

	if (n_update_) {
		delete [] n_update_;
	}

	if (z_update_) {
		delete [] z_update_;
	}
}

template<typename T>
bool FtrlWorker<T>::Initialize(
		FtrlParamServer<T>* param_server,
		size_t push_step,
		size_t fetch_step) {
	FtrlSolver<T>::alpha_ = param_server->alpha();
	FtrlSolver<T>::beta_ = param_server->beta();
	FtrlSolver<T>::l1_ = param_server->l1();
	FtrlSolver<T>::l2_ = param_server->l2();
	FtrlSolver<T>::feat_num_ = param_server->feat_num();
	FtrlSolver<T>::dropout_ = param_server->dropout();

	n_update_ = new T[FtrlSolver<T>::feat_num_];
	z_update_ = new T[FtrlSolver<T>::feat_num_];
	set_float_zero(n_update_, FtrlSolver<T>::feat_num_);
	set_float_zero(z_update_, FtrlSolver<T>::feat_num_);

	FtrlSolver<T>::n_ = new T[FtrlSolver<T>::feat_num_];
	FtrlSolver<T>::z_ = new T[FtrlSolver<T>::feat_num_];
	param_server->FetchParam(FtrlSolver<T>::n_, FtrlSolver<T>::z_);

	param_group_num_ = calc_group_num(FtrlSolver<T>::feat_num_);
	param_group_step_ = new size_t[param_group_num_];
	for(size_t i = 0; i < param_group_num_; ++i) param_group_step_[i] = 0;

	push_step_ = push_step;
	fetch_step_ = fetch_step;

	FtrlSolver<T>::init_ = true;
	return FtrlSolver<T>::init_;
}

template<typename T>
bool FtrlWorker<T>::Reset(FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return 0;

	param_server->FetchParam(FtrlSolver<T>::n_, FtrlSolver<T>::z_);

	for(size_t i = 0; i < param_group_num_; ++i) {
		param_group_step_[i] = 0;
	}
	return true;
}

template<typename T>
T FtrlWorker<T>::Update(
		const std::vector<std::pair<size_t, T> >& x,
		T y,
		FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return 0;

	std::vector<std::pair<size_t, T> > weights;
	std::vector<T> gradients;
	T wTx = 0.;

	for(auto& item : x) {
		if (util_greater(FtrlSolver<T>::dropout_, (T)0)) {
			T rand_prob = FtrlSolver<T>::uniform_dist_(FtrlSolver<T>::rand_generator_);
			if (rand_prob < FtrlSolver<T>::dropout_) {
				continue;
			}
		}
		size_t idx = item.first;
		if (idx >= FtrlSolver<T>::feat_num_) continue;

		T val = FtrlSolver<T>::GetWeight(idx);
		weights.push_back(std::make_pair(idx, val));
		gradients.push_back(item.second);
		wTx += val * item.second;
	}

	T pred = sigmoid(wTx);
	T grad = pred - y;
	std::transform(gradients.begin(), gradients.end(), gradients.begin(),
			std::bind1st(std::multiplies<T>(), grad));

	for(size_t k = 0; k < weights.size(); ++k) {
		size_t i = weights[k].first;
		size_t g = i / kParamGroupSize;

		if (param_group_step_[g] % fetch_step_ == 0) {
			param_server->FetchParamGroup(
				FtrlSolver<T>::n_,
				FtrlSolver<T>::z_,
				g);
		}

		T w_i = weights[k].second;
		T grad_i = gradients[k];
		T sigma = (sqrt(FtrlSolver<T>::n_[i] + grad_i * grad_i)
			- sqrt(FtrlSolver<T>::n_[i])) / FtrlSolver<T>::alpha_;
		FtrlSolver<T>::z_[i] += grad_i - sigma * w_i;
		FtrlSolver<T>::n_[i] += grad_i * grad_i;
		z_update_[i] += grad_i - sigma * w_i;
		n_update_[i] += grad_i * grad_i;

		if (param_group_step_[g] % push_step_ == 0) {
			param_server->PushParamGroup(n_update_, z_update_, g);
		}

		param_group_step_[g] += 1;
	}

	return pred;
}

template<typename T>
bool FtrlWorker<T>::PushParam(FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return false;

	for(size_t i = 0; i < param_group_num_; ++i) {
		param_server->PushParamGroup(n_update_, z_update_, i);
	}

	return true;
}


#endif // SRC_FAST_FTRL_SOLVER_H
