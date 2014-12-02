#include <unistd.h>
#include <cstdlib>
#include "common.h"
#include "ftrl_solver.h"

void print_usage(int argc, char* argv[]) {
	printf("Usage:\n");
	printf("\t%s -t test_file -m model -o output_file\n", argv[0]);
}

int main(int argc, char* argv[]) {
	int ch;
	
	std::string test_file;
	std::string model_file;
	std::string output_file;
	
	while( (ch = getopt(argc, argv, "t:m:o:h")) != -1) {
		switch(ch) {
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

	LRModel model(model_file.c_str());

	int max_line_len = 10240;
	char* line = (char *) malloc(sizeof(char) * max_line_len);
	double y = 0.;
	std::vector<std::pair<size_t, double> > x;
	FILE* fp = fopen(test_file.c_str(), "r");
	FILE* wfp = fopen(output_file.c_str(), "w");
	while(read_line(fp, line, max_line_len) != NULL) {
		if (!parse_line(line, y, x)) {
			fprintf(wfp, "0.0\n");
			continue;
		}
		double pred = model.Predict(x);
		pred = std::max(std::min(pred, 1. - 10e-15), 10e-15);
		fprintf(wfp, "%lf\n", pred);
	}

	free(line);
	fclose(fp);
	fclose(wfp);

	return 0;
}
