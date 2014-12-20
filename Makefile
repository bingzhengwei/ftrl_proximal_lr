
CC = g++
CPPFLAGS = -Wall -O3 -fPIC -std=c++11 -march=native
INCLUDES = -I.
LDFLAGS = -pthread

all: ftrl_train ftrl_predict

.cpp.o:
	$(CC) -c $^ $(INCLUDES) $(CPPFLAGS)

ftrl_train: ftrl_train.o stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

ftrl_predict: ftrl_predict.o stopwatch.o
	$(CC) -o $@ $^ $(INCLUDES) $(CPPFLAGS) $(LDFLAGS)

clean:
	rm -f *.o ftrl_train ftrl_predict
