#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#if !defined(__linux__)
#error "test_large_alloc is Linux-specific"
#endif

namespace {

constexpr std::size_t kThreshold = std::size_t{1} << 20;
std::atomic<unsigned> g_new_handler_calls{0};

void throwing_new_handler() {
    ++g_new_handler_calls;
    throw std::bad_alloc();
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct alignas(128) OverAlignedObject {
    unsigned char bytes[128];
};

void check_basic_allocations() {
    void* zero = ::operator new(0);
    require(zero != nullptr, "zero-sized allocation returned null");
    require(
        reinterpret_cast<std::uintptr_t>(zero) % alignof(std::max_align_t) == 0,
        "zero-sized ordinary allocation is misaligned");
    ::operator delete(zero);

    auto* small = static_cast<unsigned char*>(::operator new(256));
    require(
        reinterpret_cast<std::uintptr_t>(small) % alignof(std::max_align_t) ==
            0,
        "small ordinary allocation is misaligned");
    small[0] = 7;
    small[255] = 9;
    require(small[0] == 7 && small[255] == 9, "small allocation is unusable");
    ::operator delete(small, std::size_t{256});

    auto* threshold = new unsigned char[kThreshold];
    threshold[0] = 1;
    threshold[kThreshold - 1] = 2;
    require(threshold[0] == 1 && threshold[kThreshold - 1] == 2,
            "threshold-sized allocation is unusable");
    delete[] threshold;

    auto* large = new unsigned char[kThreshold + 8192];
    require(
        reinterpret_cast<std::uintptr_t>(large) % alignof(std::max_align_t) ==
            0,
        "large ordinary allocation is misaligned");
    large[0] = 3;
    large[kThreshold + 8191] = 5;
    require(large[0] == 3 && large[kThreshold + 8191] == 5,
            "large allocation is unusable");
    delete[] large;

    auto* over_aligned = new OverAlignedObject;
    require(reinterpret_cast<std::uintptr_t>(over_aligned) % 128 == 0,
            "over-aligned allocation lost its standard alignment");
    over_aligned->bytes[0] = 11;
    delete over_aligned;
}

void check_overflow_and_new_handler() {
    const std::new_handler previous =
        std::set_new_handler(throwing_new_handler);
    g_new_handler_calls = 0;
    bool failed = false;
    try {
        (void)::operator new(std::numeric_limits<std::size_t>::max());
    } catch (const std::bad_alloc&) {
        failed = true;
    }
    std::set_new_handler(previous);
    require(failed, "SIZE_MAX allocation did not fail with bad_alloc");
    require(g_new_handler_calls.load() == 1,
            "allocation failure did not invoke the installed new_handler once");
}

void check_concurrent_large_allocations() {
    constexpr unsigned kWorkers = 4;
    constexpr std::size_t kSize = kThreshold + 16384;
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    threads.reserve(kWorkers);
    for (unsigned worker = 0; worker < kWorkers; ++worker) {
        threads.emplace_back([worker, &failed, kSize] {
            try {
                auto* memory = new unsigned char[kSize];
                for (std::size_t i = 0; i < kSize; i += 4096) {
                    memory[i] = static_cast<unsigned char>(worker + i);
                }
                delete[] memory;
            } catch (...) {
                failed = true;
            }
        });
    }
    for (std::thread& thread : threads) thread.join();
    require(!failed.load(), "concurrent large allocation failed");
}

}  // namespace

int main() {
    try {
        check_basic_allocations();
        check_overflow_and_new_handler();
        check_concurrent_large_allocations();
    } catch (const std::exception& error) {
        std::cerr << "test_large_alloc: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test_large_alloc: PASS\n";
    return 0;
}
