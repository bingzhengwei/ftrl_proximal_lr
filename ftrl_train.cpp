#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include "ftrl_solver.h"
#include "util.h"
#include "common.h"
#include "stopwatch.h"

void print_usage(int argc, char* argv[]) {
	printf("Usage:\n");
	printf("\t%s -f input_file -m model [-t test_file] [-i iter] [-a alpha] [-b beta] [-d dropout] [-l l1] [-e l2] [-c]\n", argv[0]);
}

bool train(const std::string& input, const std::string& model, const std::string& test,
		double alpha, double beta, double l1, double l2,
		double dropout, size_t epoch, bool cache) {
	size_t feat_num = 0;
	int max_line_len = 10240;
	char* line = (char *) malloc(sizeof(char) * max_line_len);

	printf("training with settings:\n");
	printf("\talpha=[%lf] beta=[%lf] l1=[%lf] l2=[%lf] epoch=[%lu] dropout=[%lf]\n", alpha, beta, l1, l2, epoch, dropout);
	printf("\tinput_file=[%s] model_file=[%s] test_file=[%s]\n", input.c_str(), model.c_str(), test.c_str());

	StopWatch timer;
	std::vector<std::pair<size_t, double> > x;
	double y;
	size_t count = 0;
	FILE* fp = fopen(input.c_str(), "r");
	std::string cache_file = input + ".cache";

	auto read_feat_count = [&]() {
		while(read_line(fp, line, max_line_len) != NULL) {
			if (!parse_line(line, y, x)) continue;
			for(auto& i : x) {
				if (i.first + 1 > feat_num) feat_num = i.first + 1;
			}
			++count;
			if (count % 100000 == 0) {
				printf("loading=[%lu]\r", count);
				fflush(stdout);
			}
		}
		printf("\n\tinstances=[%lu] features=[%lu] time=[%lf]\n", count, feat_num, timer.StopTimer());
	};

	if (cache && file_exists(cache_file.c_str())) {
		FILE* cfp = fopen(cache_file.c_str(), "r");
		if (fscanf(cfp, "%lu %lu", &count, &feat_num) != 2) {
			throw std::runtime_error(std::string("Failed to load cache file"));
		}
		fclose(cfp);
		printf("\tinstances=[%lu] features=[%lu] from=cache\n", count, feat_num);
	} else {
		read_feat_count();
	}

	if (cache) {
		FILE* cfp = fopen(cache_file.c_str(), "w");
		fprintf(cfp, "%lu\n%lu\n", count, feat_num);
		fclose(cfp);
	}

	FtrlProximalTrainer learner(alpha, beta, l1, l2, feat_num, dropout);

	auto train_one_iter = [&](size_t iter) {
		rewind(fp);
		size_t line_no = 0;
		double loss = 0;
		while(read_line(fp, line, max_line_len) != NULL) {
			if (!parse_line(line, y, x)) continue;
			double pred = learner.Update(x, y);
			pred = std::max(std::min(pred, 1. - 10e-15), 10e-15);
			loss += y > 0 ? -log(pred) : -log(1. - pred);
			++line_no;

			if (line_no % 100000 == 0) {
				printf("epoch=%lu processed=[%.2f%%] time=[%lf] train-loss=%lf\r", iter, float(line_no) * 100. / float(count), timer.StopTimer(), loss / line_no);
				fflush(stdout);
			}
		}
		printf("\n");
	};

	auto predict_validation = [&] (const std::string& filename) {
		FILE* vfp = fopen(filename.c_str(), "r");
		size_t line_cnt = 0;
		double loss = 0;
		while(read_line(vfp, line, max_line_len) != NULL) {
			if (!parse_line(line, y, x)) continue;
			double pred = learner.Predict(x);
			pred = std::max(std::min(pred, 1. - 10e-15), 10e-15);
			loss += y > 0 ? -log(pred) : -log(1. - pred);
			++line_cnt;
		}

		fclose(vfp);
		if (line_cnt > 0)  loss /= line_cnt;
		return loss;
	};

	for(size_t iter = 0; iter < epoch; ++iter) {
		train_one_iter(iter);
		if (test.size() > 0) {
			double valid_loss = predict_validation(test);
			printf("validation-loss=[%lf]\n\n", valid_loss);
		}
	}

	learner.Save(model.c_str());
	std::string model_detail = model + ".detail";
	learner.SaveDetail(model_detail.c_str());

	fclose(fp);
	free((void *)line);
	return true;
}

int main(int argc, char* argv[]) {
	int ch;
	
	std::string input_file;
	std::string test_file;
	std::string model_file;

	double alpha = DEFAULT_ALPHA;
	double beta = DEFAULT_BETA;
	double l1 = DEFAULT_L1;
	double l2 = DEFAULT_L2;
	double dropout = 0;
	size_t epoch = 1;
	bool cache = false;
	
	while( (ch = getopt(argc, argv, "f:t:m:a:b:d:l:e:i:ch")) != -1) {
		switch(ch) {
		case 'f':
			input_file = optarg;
			break;
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
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
		case 'i':
			epoch = (size_t)atoi(optarg);
			break;
		case 'c':
			cache = true;
			break;
		case 'h':
		default:
			print_usage(argc, argv);
			exit(0);
		}
	}

	if (input_file.size() == 0 || model_file.size() == 0) {
		print_usage(argc, argv);
		exit(1);
	}

	train(input_file, model_file, test_file, alpha, beta, l1, l2, dropout, epoch, cache);

	return 0;
}
