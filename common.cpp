#include "common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "util.h"

char* read_line(FILE* fp, char* line, int& max_line_len) {
	int len;
	
	if (fgets(line, max_line_len, fp) == NULL) {
		return NULL;
	}

	while(strrchr(line, '\n') == NULL) {
		max_line_len *= 2;
		line = (char *) realloc(line, max_line_len * sizeof(char));
		len = (int) strlen(line);
		if (fgets(line + len, max_line_len - len, fp) == NULL) break;
	}

	return line;
}

bool parse_line(char* line, double& y, std::vector<std::pair<size_t, double> >& x) {
	char *endptr, *ptr;
	char *p = strtok_r(line," \t\n", &ptr);
	if(p == NULL) return false;

	y = strtod(p, &endptr);
	if(endptr == p || *endptr != '\0') return false;
	if (y < 0) y = 0;

	x.clear();
	while(1) {
		char *idx = strtok_r(NULL, ":", &ptr);
		char *val = strtok_r(NULL, " \t", &ptr);
		if (val == NULL) break;

		bool error_found = false;
		size_t k = (size_t) strtol(idx, &endptr, 10) - 1;
		if(endptr == idx || *endptr != '\0' || (int)k < 0) {
			error_found = true;
		}
		
		double v = strtod(val, &endptr);
		if(endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
			error_found = true;
		}
		
		if (!error_found) {
			x.push_back(std::make_pair(k, v));
		}
	}

	return true;
}

bool file_exists(const char *path) {
	/*struct stat st;
	int result = stat(path, &st);
	return result == 0;*/
	if (FILE * file = fopen(path, "r")) {
		fclose(file);
		return true;
	}
	return false;
}

