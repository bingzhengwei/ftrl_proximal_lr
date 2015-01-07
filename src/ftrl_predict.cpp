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

#include <unistd.h>
#include <cstdlib>
#include "src/file_parser.h"
#include "src/ftrl_solver.h"
#include "src/util.h"

void print_usage(int argc, char* argv[]) {
	printf("Usage:\n");
	printf("\t%s -t test_file -m model -o output_file\n", argv[0]);
}

int main(int argc, char* argv[]) {
	int ch;

	std::string test_file;
	std::string model_file;
	std::string output_file;

	while ((ch = getopt(argc, argv, "t:m:o:h")) != -1) {
		switch (ch) {
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'o':
			output_file = optarg;
			break;
		case 'h':
		default:
			print_usage(argc, argv);
			exit(0);
		}
	}

	if (test_file.size() == 0 || model_file.size() == 0 || output_file.size() == 0) {
		print_usage(argc, argv);
		exit(1);
	}

	LRModel<double> model;
	model.Initialize(model_file.c_str());

	double y = 0.;
	std::vector<std::pair<size_t, double> > x;
	FILE* wfp = fopen(output_file.c_str(), "w");
	size_t cnt = 0, correct = 0;
	double loss = 0.;
	FileParser<double> parser;
	parser.OpenFile(test_file.c_str());

	while (1) {
		bool res = parser.ReadSample(y, x);
		if (!res) break;

		double pred = model.Predict(x);
		pred = std::max(std::min(pred, 1. - 10e-15), 10e-15);
		fprintf(wfp, "%lf\n", pred);

		++cnt;
		double pred_label = 0;
		if (pred > 0.5) pred_label = 1;
		if (util_equal(pred_label, y)) ++correct;

		pred = std::max(std::min(pred, 1. - 10e-15), 10e-15);
		loss += y > 0 ? -log(pred) : -log(1. - pred);
	}

	if (cnt > 0) {
		printf("Accuracy = %.2lf%% (%zu/%zu)\n",
			static_cast<double>(correct) / cnt * 100, correct, cnt);
		printf("Log-likelihood = %lf\n", loss / cnt);
	}

	parser.CloseFile();
	fclose(wfp);

	return 0;
}
