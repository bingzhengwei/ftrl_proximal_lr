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

#ifndef SRC_FILE_PARSER_H
#define SRC_FILE_PARSER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include "src/lock.h"

template<typename T>
class FileParserBase {
public:
	FileParserBase() {}
	virtual ~FileParserBase() {}

public:
	virtual bool OpenFile(const char* path) = 0;
	virtual bool CloseFile() = 0;

	virtual bool ReadSample(T& y, std::vector<std::pair<size_t, T> >& x) = 0;
	virtual bool ReadSampleMultiThread(T& y, std::vector<std::pair<size_t, T> >& x) = 0;

public:
	static bool FileExists(const char* path);
};

// FileParser: parse training file with LIBSVM-format
template<typename T>
class FileParser : public FileParserBase<T> {
public:
	FileParser();
	virtual ~FileParser();

	virtual bool OpenFile(const char* path);
	virtual bool CloseFile();

	// Read a new line and Parse to <x, y>, thread-safe but not optimized for multi-threading
	virtual bool ReadSample(T& y, std::vector<std::pair<size_t, T> >& x);

	// Read a new line and Parse to <x, y>, with multi-threading capability
	virtual bool ReadSampleMultiThread(T& y, std::vector<std::pair<size_t, T> >& x);

	bool ParseSample(char* buf, T& y,
		std::vector<std::pair<size_t, T> >& x);

	// Read a new line using external buffer
	char* ReadLine(char *buf, size_t& buf_size);

private:
	// Read a new line using internal buffer and copy that to allocated new memory
	char* ReadLine();

	char* ReadLineImpl(char *buf, size_t& buf_size);

private:
	enum { kDefaultBufSize = 10240 };

	FILE* file_desc_;
	char* buf_;
	size_t buf_size_;

	SpinLock lock_;
};


template<typename T>
T* alloc_func(size_t size) {
	void* ptr = malloc(size * sizeof(T));
	return reinterpret_cast<T*>(ptr);
}

template<typename T>
T* realloc_func(T* buf, size_t size) {
	void* ptr = realloc(reinterpret_cast<void*>(buf), size * sizeof(T));
	return reinterpret_cast<T*>(ptr);
}


template<typename T>
bool FileParserBase<T>::FileExists(const char* path) {
	FILE *fp = fopen(path, "r");
	if (fp) {
		fclose(fp);
		return true;
	}

	return false;
}




template<typename T>
FileParser<T>::FileParser() : file_desc_(NULL), buf_(NULL), buf_size_(0) {
	buf_size_ = kDefaultBufSize;
	buf_ = alloc_func<char>(buf_size_);
}

template<typename T>
FileParser<T>::~FileParser() {
	if (file_desc_) {
		fclose(file_desc_);
	}

	if (buf_) {
		free(buf_);
	}

	buf_size_ = 0;
}

template<typename T>
bool FileParser<T>::OpenFile(const char* path) {
	file_desc_ = fopen(path, "r");

	if (!file_desc_) {
		return false;
	}

	return true;
}

template<typename T>
bool FileParser<T>::CloseFile() {
	if (file_desc_) {
		fclose(file_desc_);
		file_desc_ = NULL;
	}

	return true;
}

template<typename T>
char* FileParser<T>::ReadLineImpl(char* buf, size_t& buf_size) {
	if (!file_desc_) {
		return NULL;
	}

	if (fgets(buf, buf_size - 1, file_desc_) == NULL) {
		return NULL;
	}

	while (strrchr(buf, '\n') == NULL) {
		buf_size *= 2;
		buf = realloc_func<char>(buf, buf_size);
		size_t len = strlen(buf);
		if (fgets(buf + len, buf_size - len - 1, file_desc_) == NULL) break;
	}

	return buf;
}

template<typename T>
char* FileParser<T>::ReadLine() {
	std::lock_guard<SpinLock> lock(lock_);

	char *buf = ReadLineImpl(buf_, buf_size_);
	if (buf) {
		buf_ = buf;
		return strdup(buf);
	}

	return NULL;
}

template<typename T>
char* FileParser<T>::ReadLine(char *buf, size_t& buf_size) {
	std::lock_guard<SpinLock> lock(lock_);
	return ReadLineImpl(buf, buf_size);
}

template<typename T>
T string_to_real(const char *nptr, char **endptr);

template<>
float string_to_real<float> (const char *nptr, char **endptr) {
	return strtof(nptr, endptr);
}

template<>
double string_to_real<double> (const char *nptr, char **endptr) {
	return strtod(nptr, endptr);
}

template<typename T>
bool FileParser<T>::ParseSample(char* buf, T& y,
		std::vector<std::pair<size_t, T> >& x) {
	if (buf == NULL) return false;

	char *endptr, *ptr;
	char *p = strtok_r(buf, " \t\n", &ptr);
	if (p == NULL) return false;

	y = string_to_real<T> (p, &endptr);
	if (endptr == p || *endptr != '\0') return false;
	if (y < 0) y = 0;

	x.clear();
	// add bias term
	x.push_back(std::make_pair((size_t)0, (T)1));
	while (1) {
		char *idx = strtok_r(NULL, ":", &ptr);
		char *val = strtok_r(NULL, " \t", &ptr);
		if (val == NULL) break;

		bool error_found = false;
		size_t k = (size_t) strtol(idx, &endptr, 10);
		if (endptr == idx || *endptr != '\0' || static_cast<int>(k) < 0) {
			error_found = true;
		}

		T v = string_to_real<T> (val, &endptr);
		if (endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
			error_found = true;
		}

		if (!error_found) {
			x.push_back(std::make_pair(k, v));
		}
	}

	return true;
}

template<typename T>
bool FileParser<T>::ReadSample(T& y,
		std::vector<std::pair<size_t, T> >& x) {
	std::lock_guard<SpinLock> lock(lock_);
	char *buf = ReadLineImpl(buf_, buf_size_);
	if (!buf) return false;

	buf_ = buf;
	return ParseSample(buf, y, x);
}

template<typename T>
bool FileParser<T>::ReadSampleMultiThread(T& y,
		std::vector<std::pair<size_t, T> >& x) {
	char *buf = ReadLine();
	if (!buf) return false;

	bool suc = ParseSample(buf, y, x);
	free(buf);
	return suc;
}


#endif // SRC_FILE_PARSER_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
