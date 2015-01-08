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

#include "src/stopwatch.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

double StopWatch::ToSeconds() {
	auto duration = duration_cast<microseconds>(stop_ - start_).count();
	return static_cast<double>(duration) / static_cast<double>(1000000.0);
}

double StopWatch::ToMicroSeconds() {
	auto duration = duration_cast<microseconds>(stop_ - start_).count();
	return static_cast<double>(duration);
}

StopWatch::StopWatch() {
	StartTimer();
}

void StopWatch::StartTimer() {
	start_ = high_resolution_clock::now();
}

double StopWatch::StopTimer() {
	stop_ = high_resolution_clock::now();
	return ToSeconds();
}

double StopWatch::ElapsedTime() {
	return ToSeconds();
}

double StopWatch::ElapsedTimeMS() {
	return ToMicroSeconds();
}
/* vim: set ts=4 sw=4 tw=0 noet :*/
