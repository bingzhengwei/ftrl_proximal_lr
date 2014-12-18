#ifndef __LOCK_H__
#define __LOCK_H__

#include <atomic>
#include <mutex>

class SpinLock {
public:
	SpinLock() : flag_{ATOMIC_FLAG_INIT} {
	}

	void lock() {
		while(flag_.test_and_set(std::memory_order_acquire));
	}

	void unlock() {
		flag_.clear(std::memory_order_release);
	}


protected:
	std::atomic_flag flag_;
};

#endif // __LOCK_H__
