#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <locale>
#include "util.h"
#include "fast_ftrl_solver.h"
#include "ftrl_train.h"

using namespace std;

void print_usage() {
	printf("Usage: ./ftrl_train -f input_file -m model_file [options]\n"
		"options:\n"
		"-t test_file : set evaluation file\n"
		"--cache/-c : cache feature count and sample count of input file, default true\n"
		"--epoch iteration : set number of iteration, default 1\n"
		"--alpha alpha : set alpha param, default 0.15\n"
		"--beta beta : set beta param, default 1\n"
		"--l1 l1 : set l1 param, default 1\n"
		"--l2 l2 : set l2 param, default 1\n"
		"--dropout dropout : set dropout rate, default 0\n"
		"--sync-step step : set push/fetch step of async ftrl, default 32\n"
		"--start-from model_file : set to continue training from model_file\n"
		"--thread num : set thread num, default is hardware concurrency (set 1 will enable single thread mode)\n"
		"--double-precision : set to use double precision, default false\n"
		"--help : print this help\n"
	      );
}

int main (int argc, char* argv[]) {
	int opt;
	int opt_idx = 0;

	static struct option long_options[] = {
		{"epoch", required_argument, NULL, 'i'},
		{"alpha", required_argument, NULL, 'a'},
		{"beta", required_argument, NULL, 'b'},
		{"dropout", required_argument, NULL, 'd'},
		{"l1", required_argument, NULL, 'l'},
		{"l2", required_argument, NULL, 'e'},
		{"sync-step", required_argument, NULL, 's'},
		{"cache", no_argument, NULL, 'c'},
		{"start-from", required_argument, NULL, 'r'},
		{"thread", required_argument, NULL, 'n'},
		{"double-precision", no_argument, NULL, 'x'},
		{"help", no_argument, NULL, 'h'},
		{0, 0, 0, 0}
	};

	std::string input_file;
	std::string test_file;
	std::string model_file;
	std::string start_from_model;

	double alpha = DEFAULT_ALPHA;
	double beta = DEFAULT_BETA;
	double l1 = DEFAULT_L1;
	double l2 = DEFAULT_L2;
	double dropout = 0;

	size_t epoch = 1;
	bool cache = true;
	size_t push_step = kPushStep;
	size_t fetch_step = kFetchStep;
	size_t num_threads = 0;

	bool double_precision = false;

	while((opt = getopt_long(argc, argv, "f:t:m:ch", long_options, &opt_idx)) != -1) {
		switch(opt) {
		case 'f':
			input_file = optarg;
			break;
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'c':
			cache = true;
			break;
		case 'i':
			epoch = (size_t)atoi(optarg);
			break;
		case 'a':
			alpha = atof(optarg);
			break;
		case 'b':
			beta = atof(optarg);
			break;
		case 'd':
			dropout = atof(optarg);
			break;
		case 'l':
			l1 = atof(optarg);
			break;
		case 'e':
			l2 = atof(optarg);
			break;
		case 's':
			push_step = (size_t)atoi(optarg);
			fetch_step = push_step;
			break;
		case 'n':
			num_threads = (size_t)atoi(optarg);
			break;
		case 'x':
			double_precision = true;
			break;
		case 'r':
			start_from_model = optarg;
			break;
		case 'h':
		default:
			print_usage();
			exit(0);
		}
	}

	if (input_file.size() == 0 || model_file.size() == 0) {
		print_usage();
		exit(1);
	}

	const char* ptest_file = NULL;
	if (test_file.size() > 0) ptest_file = test_file.c_str();

	if (double_precision) {
		if (num_threads == 1) {
			FtrlTrainer<double> trainer;
			trainer.Initialize(epoch, cache);

			if (start_from_model.size() > 0) {
				trainer.Train(start_from_model.c_str(),
					model_file.c_str(), input_file.c_str(), ptest_file);
			} else {
				trainer.Train(alpha, beta, l1, l2, dropout,
					model_file.c_str(), input_file.c_str(), ptest_file);
			}
		} else {
			FastFtrlTrainer<double> trainer;
			trainer.Initialize(epoch, num_threads, cache, push_step, fetch_step);

			if (start_from_model.size() > 0) {
				trainer.Train(start_from_model.c_str(),
					model_file.c_str(), input_file.c_str(), ptest_file);
			} else {
				trainer.Train(alpha, beta, l1, l2, dropout,
					model_file.c_str(), input_file.c_str(), ptest_file);
			}
		}
	} else {
		if (num_threads == 1) {
			FtrlTrainer<float> trainer;
			trainer.Initialize(epoch, cache);

			if (start_from_model.size() > 0) {
				trainer.Train(start_from_model.c_str(),
					model_file.c_str(), input_file.c_str(), ptest_file);
			} else {
				trainer.Train(alpha, beta, l1, l2, dropout,
					model_file.c_str(), input_file.c_str(), ptest_file);
			}
		} else {
			FastFtrlTrainer<float> trainer;
			trainer.Initialize(epoch, num_threads, cache, push_step, fetch_step);

			if (start_from_model.size() > 0) {
				trainer.Train(start_from_model.c_str(),
					model_file.c_str(), input_file.c_str(), ptest_file);
			} else {
				trainer.Train(alpha, beta, l1, l2, dropout,
					model_file.c_str(), input_file.c_str(), ptest_file);
			}
		}
	}

	return 0;
}
