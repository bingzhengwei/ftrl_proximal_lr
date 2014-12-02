#ifndef __COMMON_H__
#define __COMMON_H__

#include <cstdio>
#include <vector>

char* read_line(FILE* fp, char* line, int& max_line_len);

bool parse_line(char* line, double& y, std::vector<std::pair<size_t, double> >& x);

bool file_exists(const char* path);

#endif // __COMMON_H__
