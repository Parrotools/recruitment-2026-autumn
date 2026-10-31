#include <algorithm>
#include <atomic>
#include <cstddef>

#ifndef MELSPEC_HUGE_ALLOC
#define MELSPEC_HUGE_ALLOC 0
#endif

#if MELSPEC_HUGE_ALLOC

#if !defined(__linux__)
#error "MELSPEC_HUGE_ALLOC is supported only on Linux"
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

#ifndef MELSPEC_TOUCH_THREADS
#define MELSPEC_TOUCH_THREADS 16
#endif

constexpr std::size_t kBigAllocationThreshold = std::size_t{1} << 20;
constexpr std::size_t kHugePageAlignment = std::size_t{2} << 20;
constexpr std::uint64_t kSmallAllocationTag = 0x6d656c736d616c6cull;
constexpr std::uint64_t kMappedAllocationTag = 0x6d656c7362696721ull;

struct alignas(std::max_align_t) AllocationHeader {
    std::uint64_t tag;
    void* mapping_base;
    std::size_t mapping_size;
    std::size_t requested;  // payload size asked for
};

static_assert(sizeof(AllocationHeader) % alignof(std::max_align_t) == 0,
              "ordinary new results must retain default alignment");

thread_local unsigned g_allocator_recursion = 0;

class RecursionGuard {
   public:
    RecursionGuard() {
        ++g_allocator_recursion;
    }
    ~RecursionGuard() {
        --g_allocator_recursion;
    }
    RecursionGuard(const RecursionGuard&) = delete;
    RecursionGuard& operator=(const RecursionGuard&) = delete;
};

class MappedRegion {
   public:
    MappedRegion(void* base, std::size_t size) : base_(base), size_(size) {
    }
    ~MappedRegion() {
        if (base_ != nullptr) munmap(base_, size_);
    }
    MappedRegion(const MappedRegion&) = delete;
    MappedRegion& operator=(const MappedRegion&) = delete;

    void* get() const {
        return base_;
    }
    std::size_t size() const {
        return size_;
    }
    void release() {
        base_ = nullptr;
    }

   private:
    void* base_;
    std::size_t size_;
};

void* malloc_block(std::size_t requested) noexcept {
    const std::size_t payload = std::max<std::size_t>(requested, 1);
    if (payload >
        std::numeric_limits<std::size_t>::max() - sizeof(AllocationHeader)) {
        return nullptr;
    }
    void* raw = std::malloc(sizeof(AllocationHeader) + payload);
    if (raw == nullptr) return nullptr;
    ::new (raw) AllocationHeader{kSmallAllocationTag, nullptr, 0, requested};
    return reinterpret_cast<unsigned char*>(raw) + sizeof(AllocationHeader);
}

std::size_t allocator_thread_limit() noexcept {
    const char* text = std::getenv("MELSPEC_THREADS");
    if (text == nullptr || *text == '\0')
        return std::numeric_limits<std::size_t>::max();
    std::size_t value = 0;
    for (const char* ch = text; *ch != '\0'; ++ch) {
        if (*ch < '0' || *ch > '9') return 1;
        const std::size_t digit = static_cast<std::size_t>(*ch - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10)
            return 1;
        value = value * 10 + digit;
    }
    return value == 0 ? 1 : value;
}

std::size_t allocator_usable_cpus() noexcept {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) return static_cast<std::size_t>(count);
    }
    return std::max<std::size_t>(
        1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
}

std::size_t split_start(std::size_t total, std::size_t worker,
                        std::size_t workers) noexcept {
    const std::size_t base = total / workers;
    const std::size_t extra = total % workers;
    return base * worker + std::min(worker, extra);
}

void touch_pages(unsigned char* base, std::size_t page_size,
                 std::size_t first_page, std::size_t end_page) noexcept {
    for (std::size_t page = first_page; page < end_page; ++page) {
        base[page * page_size] = 0;
    }
}

void touch_mapping(unsigned char* base, std::size_t usable_size,
                   std::size_t max_threads) noexcept {
    const long system_page_size = sysconf(_SC_PAGESIZE);
    // Linux pages are at least 4 KiB; the conservative fallback still touches
    // every page if sysconf is unexpectedly unavailable.
    const std::size_t page_size =
        system_page_size > 0 ? static_cast<std::size_t>(system_page_size)
                             : 4096;
    const std::size_t page_count = 1 + (usable_size - 1) / page_size;
    const std::size_t workers = std::max<std::size_t>(
        1, std::min<std::size_t>({allocator_usable_cpus(),
                                  allocator_thread_limit(), page_count,
                                  max_threads}));
    if (workers == 1) {
        touch_pages(base, page_size, 0, page_count);
        return;
    }

    std::vector<std::thread> helpers;
    bool thread_failure = false;
    try {
        helpers.reserve(workers - 1);
        for (std::size_t worker = 1; worker < workers; ++worker) {
            const std::size_t first = split_start(page_count, worker, workers);
            const std::size_t end =
                split_start(page_count, worker + 1, workers);
            helpers.emplace_back([=] {
                RecursionGuard guard;
                touch_pages(base, page_size, first, end);
            });
        }
    } catch (...) {
        thread_failure = true;
    }

    {
        RecursionGuard guard;
        const std::size_t first = split_start(page_count, 0, workers);
        const std::size_t end = split_start(page_count, 1, workers);
        touch_pages(base, page_size, first, end);
    }
    for (std::thread& helper : helpers) helper.join();
    if (thread_failure) {
        RecursionGuard guard;
        touch_pages(base, page_size, 0, page_count);
    }
}

// Set up before C++ static initialization (see allocator_init).
bool g_parallel_memset = true;  // MELSPEC_PAR_MEMSET=0 disables
using MemsetFunction = void* (*)(void*, int, std::size_t);
MemsetFunction g_libc_memset = nullptr;

// Runs on the main thread before any C++ static constructor (priority 101),
// so the prefetch thread in melspec.cpp already sees these settings.
__attribute__((constructor(101))) void allocator_init() {
    if (const char* text = std::getenv("MELSPEC_PAR_MEMSET"))
        g_parallel_memset = std::strcmp(text, "0") != 0;
    g_libc_memset =
        reinterpret_cast<MemsetFunction>(dlsym(RTLD_NEXT, "memset"));
}

// Asks the kernel to place the pages of [base, base + size) on `node` when it
// has free memory there (MPOL_PREFERRED falls back to other nodes otherwise).
void prefer_node(void* base, std::size_t size, int node) noexcept {
    if (node < 0 || node >= 64) return;
    constexpr int kMpolPreferred = 1;
    const unsigned long mask = 1ul << node;
    (void)syscall(SYS_mbind, base, size, kMpolPreferred, &mask,
                  sizeof(mask) * 8 + 1, 0);
}

void* mapped_block(std::size_t requested,
                   std::size_t touch_threads = MELSPEC_TOUCH_THREADS,
                   int node = -1) noexcept {
    const std::size_t payload = std::max<std::size_t>(requested, 1);
    constexpr std::size_t kHeaderSize = sizeof(AllocationHeader);
    if (payload > std::numeric_limits<std::size_t>::max() - kHeaderSize) {
        return nullptr;
    }
    const std::size_t content_size = payload + kHeaderSize;
    if (content_size >
        std::numeric_limits<std::size_t>::max() - (kHugePageAlignment - 1)) {
        return nullptr;
    }
    const std::size_t usable_size =
        ((content_size + kHugePageAlignment - 1) / kHugePageAlignment) *
        kHugePageAlignment;
    if (usable_size >
        std::numeric_limits<std::size_t>::max() - kHugePageAlignment) {
        return nullptr;
    }
    const std::size_t mapping_size = usable_size + kHugePageAlignment;

    void* raw = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return nullptr;
    MappedRegion mapping(raw, mapping_size);

    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(raw);
    if (address >
        std::numeric_limits<std::uintptr_t>::max() - (kHugePageAlignment - 1)) {
        return nullptr;
    }
    const std::uintptr_t aligned_address =
        (address + kHugePageAlignment - 1) &
        ~(static_cast<std::uintptr_t>(kHugePageAlignment) - 1);
    auto* base = reinterpret_cast<unsigned char*>(aligned_address);
    (void)madvise(base, usable_size, MADV_HUGEPAGE);
    prefer_node(base, usable_size, node);
    touch_mapping(base, usable_size, touch_threads);

    ::new (static_cast<void*>(base))
        AllocationHeader{kMappedAllocationTag, raw, mapping.size(), requested};
    void* user = base + sizeof(AllocationHeader);
    mapping.release();
    return user;
}

// Pre-faulted mappings parked by melspec_reserve_allocation_on_node(), each
// handed to the next big allocation of exactly its size. (Sizes must match
// exactly: reservations are made while main.cpp is still reading, and its
// 8-byte-per-sample signal vector is about as big as the float copy.)
constexpr std::size_t kReservationSlots = 4;
std::atomic<AllocationHeader*> g_reserved[kReservationSlots] = {};

void* take_reserved(std::size_t requested) noexcept {
    for (std::atomic<AllocationHeader*>& slot : g_reserved) {
        AllocationHeader* header = slot.load(std::memory_order_acquire);
        if (header == nullptr || requested != header->requested) continue;
        if (slot.compare_exchange_strong(header, nullptr,
                                         std::memory_order_acq_rel))
            return header + 1;
    }
    return nullptr;
}

void* allocate_once(std::size_t requested) noexcept {
    if (requested >= kBigAllocationThreshold && g_allocator_recursion == 0) {
        if (void* reserved = take_reserved(requested)) return reserved;
        RecursionGuard guard;
        if (void* mapped = mapped_block(requested)) return mapped;
    }
    return malloc_block(requested);
}

}  // namespace

// Maps and pre-faults, with touch_threads threads, a block for a big
// allocation the caller knows is coming (see melspec.cpp), so that allocation
// later costs neither page faults nor kernel zeroing. With node >= 0 the pages
// are preferably placed on that NUMA node instead of the touching threads'
// (the prefetch thread reserves on behalf of the main thread, which writes
// the blocks). Takes a free slot, else replaces the first one.
void melspec_reserve_allocation_on_node(std::size_t bytes, int node,
                                        std::size_t touch_threads) noexcept {
    if (bytes < kBigAllocationThreshold || g_allocator_recursion != 0) return;
    void* user;
    {
        RecursionGuard guard;
        user = mapped_block(bytes, touch_threads, node);
    }
    if (user == nullptr) return;
    auto* header = static_cast<AllocationHeader*>(user) - 1;
    for (std::atomic<AllocationHeader*>& slot : g_reserved) {
        AllocationHeader* empty = nullptr;
        if (slot.compare_exchange_strong(empty, header,
                                         std::memory_order_acq_rel))
            return;
    }
    if (AllocationHeader* old =
            g_reserved[0].exchange(header, std::memory_order_acq_rel))
        (void)munmap(old->mapping_base, old->mapping_size);
}

// memset of 32 MiB or more (main.cpp's vector value-initialization zeroes
// the 197 MB signal and float copy this way) is split over 16 threads unless
// MELSPEC_PAR_MEMSET=0; everything else goes to the C library's memset.
// Measured 7-17 ms faster end to end (report, round 8).
extern "C" void* memset(void* destination, int value,
                        std::size_t size) noexcept {
    constexpr std::size_t kParallelMemset = std::size_t{32} << 20;
    if (g_libc_memset == nullptr) {
        // Before allocator_init (other libraries' start-up code): bytewise,
        // through volatile so this loop is not turned back into memset.
        auto* bytes = static_cast<volatile unsigned char*>(destination);
        for (std::size_t i = 0; i < size; ++i)
            bytes[i] = static_cast<unsigned char>(value);
        return destination;
    }
    if (!g_parallel_memset || size < kParallelMemset ||
        g_allocator_recursion != 0)
        return g_libc_memset(destination, value, size);

    RecursionGuard guard;
    auto* bytes = static_cast<unsigned char*>(destination);
    const std::size_t workers = std::min<std::size_t>(
        {16, allocator_usable_cpus(), allocator_thread_limit()});
    if (workers <= 1) return g_libc_memset(destination, value, size);
    // Split on 4 KiB boundaries so no two threads share a page.
    const std::size_t pages = size / 4096;
    std::vector<std::thread> helpers;
    std::size_t started = 0;
    try {
        helpers.reserve(workers);
        for (std::size_t worker = 1; worker < workers; ++worker) {
            const std::size_t first =
                split_start(pages, worker, workers) * 4096;
            const std::size_t end =
                worker + 1 == workers
                    ? size
                    : split_start(pages, worker + 1, workers) * 4096;
            helpers.emplace_back([=] {
                RecursionGuard inner;
                g_libc_memset(bytes + first, value, end - first);
            });
            started = worker;
        }
    } catch (...) {
    }
    // The calling thread takes part 0 plus whatever was not handed out.
    const std::size_t own_end = split_start(pages, 1, workers) * 4096;
    g_libc_memset(bytes, value, own_end);
    if (started + 1 < workers) {
        const std::size_t rest =
            split_start(pages, started + 1, workers) * 4096;
        g_libc_memset(bytes + rest, value, size - rest);
    }
    for (std::thread& helper : helpers) helper.join();
    return destination;
}

void* operator new(std::size_t requested) {
    for (;;) {
        if (void* memory = allocate_once(requested)) return memory;
        const std::new_handler handler = std::get_new_handler();
        if (handler == nullptr) throw std::bad_alloc();
        handler();
    }
}

void* operator new[](std::size_t requested) {
    return ::operator new(requested);
}

void operator delete(void* pointer) noexcept {
    if (pointer == nullptr) return;
    auto* header = reinterpret_cast<AllocationHeader*>(
        static_cast<unsigned char*>(pointer) - sizeof(AllocationHeader));
    if (header->tag == kMappedAllocationTag) {
        (void)munmap(header->mapping_base, header->mapping_size);
    } else {
        std::free(header);
    }
}

void operator delete[](void* pointer) noexcept {
    ::operator delete(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    ::operator delete(pointer);
}

#endif  // MELSPEC_HUGE_ALLOC
