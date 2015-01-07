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

#ifndef SRC_FTRL_SOLVER_H
#define SRC_FTRL_SOLVER_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "src/util.h"

#define DEFAULT_ALPHA 0.15
#define DEFAULT_BETA 1.
#define DEFAULT_L1 1.
#define DEFAULT_L2 1.

template<typename T>
class FtrlSolver {
public:
	FtrlSolver();

	virtual ~FtrlSolver();

	virtual bool Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout = 0);

	virtual bool Initialize(const char* path);

	virtual T Update(const std::vector<std::pair<size_t, T> >& x, T y);
	virtual T Predict(const std::vector<std::pair<size_t, T> >& x);

	virtual bool SaveModelAll(const char* path);
	virtual bool SaveModel(const char* path);
	virtual bool SaveModelDetail(const char* path);

public:
	T alpha() { return alpha_; }
	T beta() { return beta_; }
	T l1() { return l1_; }
	T l2() { return l2_; }
	size_t feat_num() { return feat_num_; }
	T dropout() { return dropout_; }

protected:
	enum {kPrecision = 8};

protected:
	T GetWeight(size_t idx);

protected:
	T alpha_;
	T beta_;
	T l1_;
	T l2_;
	size_t feat_num_;
	T dropout_;

	T * n_;
	T * z_;

	bool init_;

	std::mt19937 rand_generator_;
	std::uniform_real_distribution<T> uniform_dist_;
};



template<typename T>
FtrlSolver<T>::FtrlSolver()
: alpha_(0), beta_(0), l1_(0), l2_(0), feat_num_(0),
dropout_(0), n_(NULL), z_(NULL), init_(false),
uniform_dist_(0.0, std::nextafter(1.0, std::numeric_limits<T>::max())) {}

template<typename T>
FtrlSolver<T>::~FtrlSolver() {
	if (n_) {
		delete [] n_;
	}

	if (z_) {
		delete [] z_;
	}
}

template<typename T>
void set_float_zero(T* x, size_t n) {
	for(size_t i = 0; i < n; ++i) {
		x[i] = 0;
	}
}

template<typename T>
bool FtrlSolver<T>::Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout) {
	alpha_ = alpha;
	beta_ = beta;
	l1_ = l1;
	l2_ = l2;
	feat_num_ = n;
	dropout_ = dropout;

	n_ = new T[feat_num_];
	z_ = new T[feat_num_];
	set_float_zero(n_, n);
	set_float_zero(z_, n);
	init_ = true;
	return init_;
}

template<typename T>
bool FtrlSolver<T>::Initialize(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

	fin >> alpha_ >> beta_ >> l1_ >> l2_ >> feat_num_ >> dropout_;
	if (!fin || fin.eof()) {
		fin.close();
		return false;
	}

	n_ = new T[feat_num_];
	z_ = new T[feat_num_];

	for(size_t i = 0; i < feat_num_; ++i) {
		fin >> n_[i];
		if (!fin || fin.eof()) {
			fin.close();
			return false;
		}
	}

	for(size_t i = 0; i < feat_num_; ++i) {
		fin >> z_[i];
		if (!fin || fin.eof()) {
			fin.close();
			return false;
		}
	}

	fin.close();
	init_ = true;
	return init_;
}

template<typename T>
T FtrlSolver<T>::GetWeight(size_t idx) {
	T sign = 1.;
	T val = 0.;
	if (z_[idx] < 0) {
		sign = -1.;
	}

	if (util_less_equal(sign * z_[idx], l1_)) {
		val = 0.;
	} else {
		val = (sign * l1_ - z_[idx]) / ((beta_ + sqrt(n_[idx])) / alpha_ + l2_);
	}

	return val;
}

template<typename T>
T FtrlSolver<T>::Update(const std::vector<std::pair<size_t, T> >& x, T y) {
	if (!init_) return 0;

	std::vector<std::pair<size_t, T> > weights;
	std::vector<T> gradients;
	T wTx = 0.;

	for(auto& item : x) {
		if (util_greater(dropout_, (T)0)) {
			T rand_prob = uniform_dist_(rand_generator_);
			if (rand_prob < dropout_) {
				continue;
			}
		}
		size_t idx = item.first;
		if (idx >= feat_num_) continue;

		T val = GetWeight(idx);
		weights.push_back(std::make_pair(idx, val));
		gradients.push_back(item.second);
		wTx += val * item.second;
	}

	T pred = sigmoid(wTx);
	T grad = pred - y;
	std::transform(
		gradients.begin(),
		gradients.end(),
		gradients.begin(),
		std::bind1st(std::multiplies<T>(), grad));

	for(size_t k = 0; k < weights.size(); ++k) {
		size_t i = weights[k].first;
		T w_i = weights[k].second;
		T grad_i = gradients[k];
		T sigma = (sqrt(n_[i] + grad_i * grad_i) - sqrt(n_[i])) / alpha_;
		z_[i] += grad_i - sigma * w_i;
		n_[i] += grad_i * grad_i;
	}

	return pred;
}

template<typename T>
T FtrlSolver<T>::Predict(const std::vector<std::pair<size_t, T> >& x) {
	if (!init_) return 0;

	T wTx = 0.;
	for(auto& item : x) {
		size_t idx = item.first;
		T val = GetWeight(idx);
		wTx += val * item.second;
	}

	T pred = sigmoid(wTx);
	return pred;
}

template<typename T>
bool FtrlSolver<T>::SaveModel(const char* path) {
	if (!init_) return false;

	std::fstream fout;
	std::ios_base::sync_with_stdio(false);
	fout.open(path, std::ios::out);

	if (!fout.is_open()) {
		return false;
	}

	fout << std::fixed << std::setprecision(kPrecision);
	for(size_t i = 0; i < feat_num_; ++i) {
		T w = GetWeight(i);
		fout << w << "\n";
	}

	fout.close();
	return true;
}

template<typename T>
bool FtrlSolver<T>::SaveModelDetail(const char* path) {
	if (!init_) return false;

	std::fstream fout;
	std::ios_base::sync_with_stdio(false);
	fout.open(path, std::ios::out);

	if (!fout.is_open()) {
		return false;
	}

	fout << std::fixed << std::setprecision(kPrecision);
	fout << alpha_ << "\t" << beta_ << "\t" << l1_ << "\t"
		<< l2_ << "\t" << feat_num_ << "\t" << dropout_ << "\n";

	for(size_t i = 0; i < feat_num_; ++i) {
		fout << n_[i] << "\n";
	}

	for(size_t i = 0; i < feat_num_; ++i) {
		fout << z_[i] << "\n";
	}

	fout.close();
	return true;
}

template<typename T>
bool FtrlSolver<T>::SaveModelAll(const char* path) {
	std::string model_detail = std::string(path) + ".save";
	return SaveModel(path) && SaveModelDetail(model_detail.c_str());
}



template<typename T>
class LRModel {
public:
	LRModel();
	virtual ~LRModel();

	bool Initialize(const char* path);

	T Predict(const std::vector<std::pair<size_t, T> >& x);
private:
	std::vector<T> model_;
	bool init_;
};

template<typename T>
LRModel<T>::LRModel() : init_(false) {}

template<typename T>
LRModel<T>::~LRModel() {}

template<typename T>
bool LRModel<T>::Initialize(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

	T w;
	while (fin >> w) {
		model_.push_back(w);
	}

	fin.close();

	init_ = true;
	return init_;
}

template<typename T>
T LRModel<T>::Predict(const std::vector<std::pair<size_t, T> >& x) {
	if (!init_) return 0;

	T wTx = 0.;
	for(auto& item : x) {
		if (item.first >= model_.size()) continue;
		wTx += model_[item.first] * item.second;
	}
	T pred = sigmoid(wTx);
	return pred;
}

#endif // SRC_FTRL_SOLVER_H
