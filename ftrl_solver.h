#ifndef __FTRL_TRAIN_H__
#define __FTRL_TRAIN_H__

#include <cstdlib>
#include <vector>
#include <random>

#define DEFAULT_ALPHA 0.15
#define DEFAULT_BETA 1.
#define DEFAULT_L1 1.
#define DEFAULT_L2 1.

class FtrlProximalTrainer {
public:
	FtrlProximalTrainer(double alpha, double beta, double l1,
			double l2, size_t n, double dropout = 0);
	FtrlProximalTrainer(const char* path);

	virtual ~FtrlProximalTrainer();

	double Update(std::vector<std::pair<size_t, double> >& x, double y);
	double Predict(std::vector<std::pair<size_t, double> >& x);

	bool Save(const char* path);
	bool SaveDetail(const char* path);

private:
	double alpha_;
	double beta_;
	double l1_;
	double l2_;
	size_t feat_num_;
	double dropout_;

	double * n_;
	double * z_;

	std::mt19937 rand_generator_;
	std::uniform_real_distribution<double> uniform_dist_;
};

class LRModel {
public:
	LRModel(const char* path);
	virtual ~LRModel();

	double Predict(std::vector<std::pair<size_t, double> >& x);
private:
	std::vector<double> model_;
};

#endif // __FTRL_TRAIN_H__
