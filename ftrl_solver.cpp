#include "ftrl_solver.h"
#include <cstdio>
#include "util.h"

FtrlPrimalTrainer::FtrlPrimalTrainer(double alpha, double beta, double l1,
		double l2, size_t n, double dropout)
: alpha_(alpha), beta_(beta), l1_(l1), l2_(l2), feat_num_(n), dropout_(dropout),
uniform_dist_(0.0, std::nextafter(1.0, std::numeric_limits<double>::max())) {
	n_ = sse_malloc(feat_num_);
	z_ = sse_malloc(feat_num_);
	vector_set(n_, 0, feat_num_);
	vector_set(z_, 0, feat_num_);
}

FtrlPrimalTrainer::FtrlPrimalTrainer(const char* path)
: uniform_dist_(0.0, std::nextafter(1.0, std::numeric_limits<double>::max())) {
	FILE* fp = fopen(path, "r");

	fscanf(fp, "%lf", &alpha_);
	fscanf(fp, "%lf", &beta_);
	fscanf(fp, "%lf", &l1_);
	fscanf(fp, "%lf", &l2_);
	fscanf(fp, "%lu", &feat_num_);
	fscanf(fp, "%lf", &dropout_);

	n_ = sse_malloc(feat_num_);
	z_ = sse_malloc(feat_num_);
	vector_set(n_, 0, feat_num_);
	vector_set(z_, 0, feat_num_);

	double v;
	for(size_t i = 0; i < feat_num_; ++i) {
		fscanf(fp, "%lf", &v);
		n_[i] = v;
	}

	for(size_t i = 0; i < feat_num_; ++i) {
		fscanf(fp, "%lf", &v);
		z_[i] = v;
	}

	fclose(fp);
}

FtrlPrimalTrainer::~FtrlPrimalTrainer() {
	sse_free(n_);
	sse_free(z_);
}

double FtrlPrimalTrainer::Update(std::vector<std::pair<size_t, double> >& x, double y) {
	std::vector<std::pair<size_t, double> > weights;
	double wTx = 0.;

	for(auto& item : x) {
		if (double_greater(dropout_, 0)) {
			double rand_prob = uniform_dist_(rand_generator_);
			if (rand_prob < dropout_) {
				continue;
			}
		}
		double sign = 1.;	
		size_t idx = item.first;
		double val = 0.;
		if (z_[idx] < 0) {
			sign = -1.;
		}

		if (double_less_equal(sign * z_[idx], l1_)) {
			val = 0.;
		} else {
			val = (sign * l1_ - z_[idx]) / ((beta_ + sqrt(n_[idx])) / alpha_ + l2_);
		}
		weights.push_back(std::make_pair(idx, val));
		wTx += val * item.second;
	}

	double pred = 1. / (1. + safe_exp(-wTx));
	double grad = pred - y;

	for(auto& item : weights) {
		size_t i = item.first;
		double sigma = (sqrt(n_[i] + grad * grad) - sqrt(n_[i])) / alpha_;
		z_[i] += grad - sigma * item.second;
		n_[i] += grad * grad;
	}

	return pred;
}

double FtrlPrimalTrainer::Predict(std::vector<std::pair<size_t, double> >& x) {
	double wTx = 0.;

	for(auto& item : x) {
		double sign = 1.;	
		size_t idx = item.first;
		double val = 0.;
		if (z_[idx] < 0) {
			sign = -1.;
		}

		if (double_less_equal(sign * z_[idx], l1_)) {
			val = 0.;
		} else {
			val = (sign * l1_ - z_[idx]) / ((beta_ + sqrt(n_[idx])) / alpha_ + l2_);
		}
		wTx += val * item.second;
	}

	double pred = 1. / (1. + safe_exp(-wTx));
	return pred;
}

bool FtrlPrimalTrainer::Save(const char* path) {
	FILE* fp = fopen(path, "w");


	for(size_t i = 0; i < feat_num_; ++i) {
		double w = 0.;
		double sign = 1.;	
		if (z_[i] < 0) {
			sign = -1.;
		}
		if (double_less_equal(sign * z_[i], l1_)) {
			w = 0.;
		} else {
			w = (sign * l1_ - z_[i]) / ((beta_ + sqrt(n_[i])) / alpha_ + l2_);
		}
		fprintf(fp, "%lf\n", w);
	}

	fclose(fp);
	return true;
}

bool FtrlPrimalTrainer::SaveDetail(const char* path) {
	FILE* fp = fopen(path, "w");
	fprintf(fp, "%lf\n", alpha_);
	fprintf(fp, "%lf\n", beta_);
	fprintf(fp, "%lf\n", l1_);
	fprintf(fp, "%lf\n", l2_);
	fprintf(fp, "%lu\n", feat_num_);
	fprintf(fp, "%lf\n", dropout_);

	for(size_t i = 0; i < feat_num_; ++i) {
		fprintf(fp, "%lf\n", n_[i]);
	}

	for(size_t i = 0; i < feat_num_; ++i) {
		fprintf(fp, "%lf\n", z_[i]);
	}

	fclose(fp);
	return true;
}

LRModel::LRModel(const char* path) {
	FILE* fp = fopen(path, "r");

	double w = 0;
	while( fscanf(fp, "%lf", &w) == 1) {
		model_.push_back(w);
	}

	fclose(fp);
}

LRModel::~LRModel() {
}

double LRModel::Predict(std::vector<std::pair<size_t, double> >& x) {
	double wTx = 0.;
	for(auto& item : x) {
		if (item.first >= model_.size()) continue;
		wTx += model_[item.first] * item.second;
	}
	double pred = 1. / (1. + safe_exp(-wTx));
	return pred;
} 


