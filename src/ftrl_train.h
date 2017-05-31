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

#ifndef SRC_FTRL_TRAIN_H
#define SRC_FTRL_TRAIN_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include "src/fast_ftrl_solver.h"
#include "src/file_parser.h"
#include "src/ftrl_solver.h"
#include "src/stopwatch.h"

template<typename T>
size_t read_problem_info(
	const char* train_file,
	bool read_cache,
	size_t& line_cnt,
	size_t num_threads = 0);

template<typename T, class Func>
T evaluate_file(const char* path, const Func& func_predict, size_t num_threads = 0);

template<typename T>
T calc_loss(T y, T pred) {
	T max_sigmoid = static_cast<T>(MAX_SIGMOID);
	T min_sigmoid = static_cast<T>(MIN_SIGMOID);
	T one = 1.;
	pred = std::max(std::min(pred, max_sigmoid), min_sigmoid);
	T loss = y > 0 ? -log(pred) : -log(std::max(one - pred, min_sigmoid));
	return loss;
}

template<typename T>
class FtrlTrainer {
public:
	FtrlTrainer();

	virtual ~FtrlTrainer();

	bool Initialize(size_t epoch, bool cache_feature_num = true);

	bool Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
        size_t feat_num,
		const char* model_file,
		const char* train_file,
		const char* test_file = NULL);

	bool Train(
		const char* last_model,
		const char* model_file,
		const char* train_file,
		const char* test_file = NULL);

protected:
	bool TrainImpl(
		const char* model_file,
		const char* train_file,
		size_t line_cnt,
		const char* test_file = NULL);

private:
	size_t epoch_;
	bool cache_feature_num_;
	FtrlSolver<T> solver_;
	bool init_;
    bool read_stdin_;
};

template<typename T>
class LockFreeFtrlTrainer {
public:
	LockFreeFtrlTrainer();

	virtual ~LockFreeFtrlTrainer();

	bool Initialize(
		size_t epoch,
		size_t num_threads,
		bool cache_feature_num = true);

	bool Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
		const char* model_file,
		const char* train_file,
		const char* test_file = NULL);

	bool Train(
		const char* last_model,
		const char* model_file,
		const char* train_file,
		const char* test_file = NULL);

protected:
	bool TrainImpl(
		const char* model_file,
		const char* train_file,
		size_t line_cnt,
		const char* test_file = NULL);

private:
	size_t epoch_;
	bool cache_feature_num_;
	FtrlSolver<T> solver_;
	size_t num_threads_;
	bool init_;
};

template<typename T>
class FastFtrlTrainer {
public:
	FastFtrlTrainer();

	virtual ~FastFtrlTrainer();

	bool Initialize(
		size_t epoch,
		size_t num_threads = 0,
		bool cache_feature_num = true,
		T burn_in = 0,
		size_t push_step = kPushStep,
		size_t fetch_step = kFetchStep);

	bool Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
		const char* model_file,
		const char* train_file,
		const char* test_file = NULL);

	bool Train(
		const char* last_model,
		const char* model_file,
		const char* train_file,
		const char* test_file = NULL);

protected:
	bool TrainImpl(
		const char* model_file,
		const char* train_file,
		size_t line_cnt,
		const char* test_file = NULL);

private:
	size_t epoch_;
	bool cache_feature_num_;
	size_t push_step_;
	size_t fetch_step_;
	T burn_in_;

	FtrlParamServer<T> param_server_;
	size_t num_threads_;

	bool init_;
};



template<typename T>
FtrlTrainer<T>::FtrlTrainer()
: epoch_(0), cache_feature_num_(false), init_(false), read_stdin_(false) { }

template<typename T>
FtrlTrainer<T>::~FtrlTrainer() {
}

template<typename T>
bool FtrlTrainer<T>::Initialize(size_t epoch, bool cache_feature_num) {
	epoch_ = epoch;
	cache_feature_num_ = cache_feature_num;

	init_ = true;
	return init_;
}

template<typename T>
bool FtrlTrainer<T>::Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
        size_t feat_num,
		const char* model_file,
		const char* train_file,
		const char* test_file) {
	if (!init_) return false;
    if (strcmp(train_file, "stdin") == 0) {
        read_stdin_ = true;
        epoch_ = 1;
        cache_feature_num_ = false;
    }
	size_t line_cnt = 0;
    if (!read_stdin_) {
	    feat_num = read_problem_info<T>(train_file, cache_feature_num_, line_cnt);
    }
	if (feat_num == 0) {
	    printf("Usage: ./ftrl_train -f input_file -m model_file [options]\n"
		    "options:\n"
		    "--feat-num num : when use stdin as input_file, set feature num, default is 0\n"
        );
        return false;
    }

	if (!solver_.Initialize(alpha, beta, l1, l2, feat_num, dropout)) {
		return false;
	}

	return TrainImpl(model_file, train_file, line_cnt, test_file);
}

template<typename T>
bool FtrlTrainer<T>::Train(
		const char* last_model,
		const char* model_file,
		const char* train_file,
		const char* test_file) {
	if (!init_) return false;
    if (strcmp(train_file, "stdin") == 0) {
        read_stdin_ = true;
        epoch_ = 1;
        cache_feature_num_ = false;
    }

	size_t line_cnt = 0;
	if (!read_stdin_) {
		size_t feat_num = read_problem_info<T>(train_file, cache_feature_num_, line_cnt);
		if (feat_num == 0) return false;
	}

	if (!solver_.Initialize(last_model)) {
		return false;
	}

	return TrainImpl(model_file, train_file, line_cnt, test_file);
}

template<typename T>
bool FtrlTrainer<T>::TrainImpl(
		const char* model_file,
		const char* train_file,
		size_t line_cnt,
		const char* test_file) {
	if (!init_) return false;

	fprintf(
		stdout,
		"params={alpha:%.2f, beta:%.2f, l1:%.2f, l2:%.2f, dropout:%.2f, epoch:%zu}\n",
		static_cast<float>(solver_.alpha()),
		static_cast<float>(solver_.beta()),
		static_cast<float>(solver_.l1()),
		static_cast<float>(solver_.l2()),
		static_cast<float>(solver_.dropout()),
		epoch_);

	auto predict_func = [&] (const std::vector<std::pair<size_t, T> >& x) {
		return solver_.Predict(x);
	};

	StopWatch timer;
	double last_time = 0;
	for (size_t iter = 0; iter < epoch_; ++iter) {
		FileParser<T> file_parser;
		file_parser.OpenFile(train_file);
		std::vector<std::pair<size_t, T> > x;
		T y;

		size_t cur_cnt = 0, last_cnt = 0;
		T loss = 0;
		while (file_parser.ReadSample(y, x)) {
			T pred = solver_.Update(x, y);
			loss += calc_loss(y, pred);
			++cur_cnt;

			if (cur_cnt - last_cnt > 100000 && timer.StopTimer() - last_time > 0.5) {
                if (!read_stdin_ && line_cnt > 0) {
                    fprintf(
                        stdout,
                        "epoch=%zu processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\r",
                        iter,
                        cur_cnt * 100 / static_cast<float>(line_cnt),
                        timer.ElapsedTime(),
                        static_cast<float>(loss) / cur_cnt);
                }
                else {
                    fprintf(
                        stdout,
                        "epoch=%zu processed=[%zu] time=[%.2f] train-loss=[%.6f]\r",
                        iter,
                        cur_cnt,
                        timer.ElapsedTime(),
                        static_cast<float>(loss) / cur_cnt);
                }
				fflush(stdout);
				last_cnt = cur_cnt;
				last_time = timer.ElapsedTime();
			}
		}

        if (!read_stdin_ && line_cnt > 0) {
            fprintf(
                stdout,
                "epoch=%zu processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\n",
                iter,
                cur_cnt * 100 / static_cast<float>(line_cnt),
                timer.ElapsedTime(),
                static_cast<float>(loss) / cur_cnt);
        }
        else {
            fprintf(
                stdout,
                "epoch=%zu processed=[%zu] time=[%.2f] train-loss=[%.6f]\n",
                iter,
                cur_cnt,
                timer.ElapsedTime(),
                static_cast<float>(loss) / cur_cnt);
        }
		file_parser.CloseFile();

		if (test_file) {
			T eval_loss = evaluate_file<T>(test_file, predict_func);
			printf("validation-loss=[%lf]\n", static_cast<double>(eval_loss));
		}
	}

	return solver_.SaveModelAll(model_file);
}



template<typename T>
LockFreeFtrlTrainer<T>::LockFreeFtrlTrainer()
: epoch_(0), cache_feature_num_(false), num_threads_(0), init_(false) { }

template<typename T>
LockFreeFtrlTrainer<T>::~LockFreeFtrlTrainer() {
}

template<typename T>
bool LockFreeFtrlTrainer<T>::Initialize(
		size_t epoch,
		size_t num_threads,
		bool cache_feature_num) {
	epoch_ = epoch;
	cache_feature_num_ = cache_feature_num;
	num_threads_ = num_threads;

	init_ = true;
	return init_;
}

template<typename T>
bool LockFreeFtrlTrainer<T>::Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
		const char* model_file,
		const char* train_file,
		const char* test_file) {
	if (!init_) return false;

	size_t line_cnt = 0;
	size_t feat_num = read_problem_info<T>(train_file, cache_feature_num_, line_cnt, num_threads_);
	if (feat_num == 0) return false;

	if (!solver_.Initialize(alpha, beta, l1, l2, feat_num, dropout)) {
		return false;
	}

	return TrainImpl(model_file, train_file, line_cnt, test_file);
}

template<typename T>
bool LockFreeFtrlTrainer<T>::Train(
		const char* last_model,
		const char* model_file,
		const char* train_file,
		const char* test_file) {
	if (!init_) return false;

	size_t line_cnt = 0;
	size_t feat_num = read_problem_info<T>(train_file, cache_feature_num_, line_cnt, num_threads_);
	if (feat_num == 0) return false;

	if (!solver_.Initialize(last_model)) {
		return false;
	}

	return TrainImpl(model_file, train_file, line_cnt, test_file);
}

template<typename T>
bool LockFreeFtrlTrainer<T>::TrainImpl(
		const char* model_file,
		const char* train_file,
		size_t line_cnt,
		const char* test_file) {
	if (!init_) return false;

	fprintf(
		stdout,
		"params={alpha:%.2f, beta:%.2f, l1:%.2f, l2:%.2f, dropout:%.2f, epoch:%zu}\n",
		static_cast<float>(solver_.alpha()),
		static_cast<float>(solver_.beta()),
		static_cast<float>(solver_.l1()),
		static_cast<float>(solver_.l2()),
		static_cast<float>(solver_.dropout()),
		epoch_);

	auto predict_func = [&] (const std::vector<std::pair<size_t, T> >& x) {
		return solver_.Predict(x);
	};

	StopWatch timer;
	for (size_t iter = 0; iter < epoch_; ++iter) {
		FileParser<T> file_parser;
		file_parser.OpenFile(train_file);

		size_t count = 0;
		T loss = 0;

		SpinLock lock;
		auto worker_func = [&] (size_t i) {
			std::vector<std::pair<size_t, T> > x;
			T y;
			size_t local_count = 0;
			T local_loss = 0;
			while (1) {
				if (!file_parser.ReadSampleMultiThread(y, x)) {
					break;
				}

				T pred = solver_.Update(x, y);
				local_loss += calc_loss(y, pred);
				++local_count;

				if (i == 0 && local_count % 10000 == 0) {
					size_t tmp_cnt = std::min(local_count * num_threads_, line_cnt);
					fprintf(
						stdout,
						"epoch=%zu processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\r",
						iter,
						tmp_cnt * 100 / static_cast<float>(line_cnt),
						timer.StopTimer(),
						static_cast<float>(local_loss) / local_count);
					fflush(stdout);
				}
			} {
				std::lock_guard<SpinLock> lockguard(lock);
				count += local_count;
				loss += local_loss;
			}
		};

		util_parallel_run(worker_func, num_threads_);

		file_parser.CloseFile();

		fprintf(
			stdout,
			"epoch=%zu processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\n",
			iter,
			count * 100 / static_cast<float>(line_cnt),
			timer.StopTimer(),
			static_cast<float>(loss) / count);

		if (test_file) {
			T eval_loss = evaluate_file<T>(test_file, predict_func);
			printf("validation-loss=[%lf]\n", static_cast<double>(eval_loss));
		}
	}

	return solver_.SaveModelAll(model_file);
}



template<typename T>
FastFtrlTrainer<T>::FastFtrlTrainer()
: epoch_(0), cache_feature_num_(false), push_step_(0),
fetch_step_(0), param_server_(), num_threads_(0), init_(false) { }

template<typename T>
FastFtrlTrainer<T>::~FastFtrlTrainer() {
}

template<typename T>
bool FastFtrlTrainer<T>::Initialize(
		size_t epoch,
		size_t num_threads,
		bool cache_feature_num,
		T burn_in,
		size_t push_step,
		size_t fetch_step) {
	epoch_ = epoch;
	cache_feature_num_ = cache_feature_num;
	push_step_ = push_step;
	fetch_step_ = fetch_step;
	if (num_threads == 0) {
		num_threads_ = std::thread::hardware_concurrency();
	} else {
		num_threads_ = num_threads;
	}

	burn_in_ = burn_in;

	init_ = true;
	return init_;
}

template<typename T>
size_t read_problem_info(
		const char* train_file,
		bool read_cache,
		size_t& line_cnt,
		size_t num_threads) {
	size_t feat_num = 0;
	line_cnt = 0;

	SpinLock lock;
	FileParser<T> parser;

	auto read_from_cache = [&](const char* path) {
		std::fstream fin;
		fin.open(path, std::ios::in);
		fin >> line_cnt >> feat_num;
		if (!fin || fin.eof()) {
			feat_num = 0;
			line_cnt = 0;
		}
		fin.close();
	};

	auto write_to_cache = [&](const char* path) {
		std::fstream fout;
		fout.open(path, std::ios::out);
		fout << line_cnt << "\t" << feat_num << "\n";
		fout.close();
	};

	auto read_problem_worker = [&](size_t i) {
		size_t local_max_feat = 0;
		size_t local_count = 0;
		std::vector<std::pair<size_t, T> > local_x;
		T local_y;
		while (1) {
			if (!parser.ReadSampleMultiThread(local_y, local_x)) break;
			for (auto& item : local_x) {
				if (item.first + 1 > local_max_feat) local_max_feat = item.first + 1;
			}
			++local_count;
		} {
			std::lock_guard<SpinLock> lockguard(lock);
			line_cnt += local_count;
			if (local_max_feat > feat_num) feat_num = local_max_feat;
		}
	};

	std::string cache_file = std::string(train_file) + ".cache";
	bool cache_exists = FileParserBase<T>::FileExists(cache_file.c_str());
	if (read_cache && cache_exists) {
		read_from_cache(cache_file.c_str());
	} else {
		parser.OpenFile(train_file);
		fprintf(stdout, "loading...");
		fflush(stdout);
		util_parallel_run(read_problem_worker, num_threads);
		parser.CloseFile();
	}

	fprintf(stdout, "\rinstances=[%zu] features=[%zu]\n", line_cnt, feat_num);

	if (read_cache && !cache_exists) {
		write_to_cache(cache_file.c_str());
	}

	return feat_num;
}

template<typename T>
bool FastFtrlTrainer<T>::Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
		const char* model_file,
		const char* train_file,
		const char* test_file) {
	if (!init_) return false;

	size_t line_cnt = 0;
	size_t feat_num = read_problem_info<T>(train_file, cache_feature_num_, line_cnt, num_threads_);
	if (feat_num == 0) return false;

	if (!param_server_.Initialize(alpha, beta, l1, l2, feat_num, dropout)) {
		return false;
	}

	return TrainImpl(model_file, train_file, line_cnt, test_file);
}

template<typename T>
bool FastFtrlTrainer<T>::Train(
		const char* last_model,
		const char* model_file,
		const char* train_file,
		const char* test_file) {
	if (!init_) return false;

	size_t line_cnt = 0;
	size_t feat_num = read_problem_info<T>(train_file, cache_feature_num_, line_cnt, num_threads_);
	if (feat_num == 0) return false;

	if (!param_server_.Initialize(last_model)) {
		return false;
	}

	return TrainImpl(model_file, train_file, line_cnt, test_file);
}

template<typename T>
bool FastFtrlTrainer<T>::TrainImpl(
		const char* model_file,
		const char* train_file,
		size_t line_cnt,
		const char* test_file) {
	if (!init_) return false;

	fprintf(
		stdout,
		"params={alpha:%.2f, beta:%.2f, l1:%.2f, l2:%.2f, dropout:%.2f, epoch:%zu}\n",
		static_cast<float>(param_server_.alpha()),
		static_cast<float>(param_server_.beta()),
		static_cast<float>(param_server_.l1()),
		static_cast<float>(param_server_.l2()),
		static_cast<float>(param_server_.dropout()),
		epoch_);

	FtrlWorker<T>* solvers = new FtrlWorker<T>[num_threads_];
	for (size_t i = 0; i < num_threads_; ++i) {
		solvers[i].Initialize(&param_server_, push_step_, fetch_step_);
	}

	auto predict_func = [&] (const std::vector<std::pair<size_t, T> >& x) {
		return param_server_.Predict(x);
	};

	StopWatch timer;
	for (size_t iter = 0; iter < epoch_; ++iter) {
		FileParser<T> file_parser;
		file_parser.OpenFile(train_file);
		size_t count = 0;
		T loss = 0;

		SpinLock lock;
		auto worker_func = [&] (size_t i) {
			std::vector<std::pair<size_t, T> > x;
			T y;
			size_t local_count = 0;
			T local_loss = 0;
			while (1) {
				if (!file_parser.ReadSampleMultiThread(y, x)) {
					break;
				}

				T pred = solvers[i].Update(x, y, &param_server_);
				local_loss += calc_loss(y, pred);
				++local_count;

				if (i == 0 && local_count % 10000 == 0) {
					size_t tmp_cnt = std::min(local_count * num_threads_, line_cnt);
					fprintf(
						stdout,
						"epoch=%zu processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\r",
						iter,
						tmp_cnt * 100 / static_cast<float>(line_cnt),
						timer.StopTimer(),
						static_cast<float>(local_loss) / local_count);
					fflush(stdout);
				}
			} {
				std::lock_guard<SpinLock> lockguard(lock);
				count += local_count;
				loss += local_loss;
			}

			solvers[i].PushParam(&param_server_);
		};

		if (iter == 0 && util_greater(burn_in_, (T)0)) {
			size_t burn_in_cnt = (size_t) (burn_in_ * line_cnt);
			std::vector<std::pair<size_t, T> > x;
			T y;
			T local_loss = 0;
			for (size_t i = 0; i < burn_in_cnt; ++i) {
				if (!file_parser.ReadSample(y, x)) {
					break;
				}

				T pred = param_server_.Update(x, y);
				local_loss += calc_loss(y, pred);
				if (i % 10000 == 0) {
					fprintf(
						stdout,
						"burn-in processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\r",
						(i + 1) * 100 / static_cast<float>(line_cnt),
						timer.StopTimer(),
						static_cast<float>(local_loss) / (i + 1));
					fflush(stdout);
				}
			}

			fprintf(
				stdout,
				"burn-in processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\n",
				burn_in_cnt * 100 / static_cast<float>(line_cnt),
				timer.StopTimer(),
				static_cast<float>(local_loss) / burn_in_cnt);

			if (util_equal(burn_in_, (T)1)) continue;
		}

		for (size_t i = 0; i < num_threads_; ++i) {
			solvers[i].Reset(&param_server_);
		}

		util_parallel_run(worker_func, num_threads_);

		file_parser.CloseFile();

		fprintf(
			stdout,
			"epoch=%zu processed=[%.2f%%] time=[%.2f] train-loss=[%.6f]\n",
			iter,
			count * 100 / static_cast<float>(line_cnt),
			timer.StopTimer(),
			static_cast<float>(loss) / count);

		if (test_file) {
			T eval_loss = evaluate_file<T>(test_file, predict_func, num_threads_);
			printf("validation-loss=[%lf]\n", static_cast<double>(eval_loss));
		}
	}

	delete [] solvers;
	return param_server_.SaveModelAll(model_file);
}

template<typename T, class Func>
T evaluate_file(const char* path, const Func& func_predict, size_t num_threads) {
	FileParser<T> parser;
	parser.OpenFile(path);

	size_t count = 0;
	T loss = 0;
	SpinLock lock;
	auto predict_worker = [&](size_t i) {
		size_t local_count = 0;
		T local_loss = 0;
		std::vector<std::pair<size_t, T> > local_x;
		T local_y;
		while (1) {
			bool res = parser.ReadSampleMultiThread(local_y, local_x);
			if (!res) break;

			local_loss += calc_loss(local_y, func_predict(local_x));
			++local_count;
		} {
			std::lock_guard<SpinLock> lockguard(lock);
			count += local_count;
			loss += local_loss;
		}
	};

	util_parallel_run(predict_worker, num_threads);

	parser.CloseFile();
	if (count > 0)  loss /= count;
	return loss;
}


#endif // SRC_FTRL_TRAIN_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
