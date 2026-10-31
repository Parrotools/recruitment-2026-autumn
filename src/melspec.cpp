#include "melspec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <complex>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#ifndef MELSPEC_SCALAR
#define MELSPEC_SCALAR 0
#endif

#ifndef MELSPEC_FUSED_FFT
#define MELSPEC_FUSED_FFT 0
#endif

#ifndef MELSPEC_AVX512
#define MELSPEC_AVX512 0
#endif

#ifndef MELSPEC_MKL
#define MELSPEC_MKL 0
#endif

#if MELSPEC_MKL
#include <mkl_dfti.h>
#endif

#ifndef MELSPEC_CUDA
#define MELSPEC_CUDA 0
#endif

#if MELSPEC_CUDA
#include "melspec_cuda.h"
#endif

#if defined(__linux__) && defined(MELSPEC_HUGE_ALLOC) && MELSPEC_HUGE_ALLOC
// Defined in large_alloc.cpp; weak so builds without the allocator (such as
// the unit test) link, with the address then being null.
__attribute__((weak)) void melspec_reserve_allocation(
    std::size_t bytes) noexcept;
#define MELSPEC_HAS_RESERVE 1
#else
#define MELSPEC_HAS_RESERVE 0
#endif

#if MELSPEC_AVX512 && !MELSPEC_SCALAR && !defined(__AVX512F__)
#error \
    "MELSPEC_AVX512=1 requires AVX-512F compiler support (for example -mavx512f)"
#endif

#if MELSPEC_SCALAR
constexpr std::size_t kFrameLanes = 1;
using FrameVec = double;
static FrameVec make_frame_vec(const double* lanes) {
    return lanes[0];
}
static double frame_lane(FrameVec value, std::size_t) {
    return value;
}
static FrameVec zero_frame_vec() {
    return 0.0;
}
#elif MELSPEC_AVX512
constexpr std::size_t kFrameLanes = 8;
using FrameVec = double __attribute__((vector_size(64)));
static FrameVec make_frame_vec(const double* lanes) {
    FrameVec result;
    for (std::size_t i = 0; i < kFrameLanes; ++i) result[i] = lanes[i];
    return result;
}
static double frame_lane(FrameVec value, std::size_t lane) {
    return value[lane];
}
static FrameVec zero_frame_vec() {
    return FrameVec{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}
#elif defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
constexpr std::size_t kFrameLanes = 4;
using FrameVec = double __attribute__((vector_size(32)));
static FrameVec make_frame_vec(const double* lanes) {
    FrameVec result;
    for (std::size_t i = 0; i < kFrameLanes; ++i) result[i] = lanes[i];
    return result;
}
static double frame_lane(FrameVec value, std::size_t lane) {
    return value[lane];
}
static FrameVec zero_frame_vec() {
    return FrameVec{0.0, 0.0, 0.0, 0.0};
}
#elif defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
constexpr std::size_t kFrameLanes = 2;
using FrameVec = double __attribute__((vector_size(16)));
static FrameVec make_frame_vec(const double* lanes) {
    FrameVec result;
    for (std::size_t i = 0; i < kFrameLanes; ++i) result[i] = lanes[i];
    return result;
}
static double frame_lane(FrameVec value, std::size_t lane) {
    return value[lane];
}
static FrameVec zero_frame_vec() {
    return FrameVec{0.0, 0.0};
}
#else
constexpr std::size_t kFrameLanes = 1;
using FrameVec = double;
static FrameVec make_frame_vec(const double* lanes) {
    return lanes[0];
}
static double frame_lane(FrameVec value, std::size_t) {
    return value;
}
static FrameVec zero_frame_vec() {
    return 0.0;
}
#endif

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Twiddles {
    std::vector<double> real;
    std::vector<double> imag;
};

Twiddles make_twiddles(std::size_t n) {
    Twiddles result{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0)};
    for (std::size_t len = 2; len <= n;) {
        const std::size_t half = len / 2;
        const double angle = -2.0 * kPi / static_cast<double>(len);
        const double root_real = std::cos(angle);
        const double root_imag = std::sin(angle);
        double w_real = 1.0;
        double w_imag = 0.0;
        for (std::size_t j = 0; j < half; ++j) {
            result.real[half + j] = w_real;
            result.imag[half + j] = w_imag;
            const double next_real = w_real * root_real - w_imag * root_imag;
            w_imag = w_real * root_imag + w_imag * root_real;
            w_real = next_real;
        }
        if (len == n) break;
        len *= 2;
    }
    return result;
}

// Optional stage timing for profiling: set MELSPEC_TRACE=1 to print the
// elapsed time of each compute stage to stderr.
class StageTrace {
   public:
    StageTrace() : enabled_(std::getenv("MELSPEC_TRACE") != nullptr) {
        last_ = std::chrono::steady_clock::now();
    }
    bool enabled() const {
        return enabled_;
    }
    void mark(const char* stage) {
        if (!enabled_) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "trace %s %.6f\n", stage,
                     std::chrono::duration<double>(now - last_).count());
        last_ = now;
    }

   private:
    bool enabled_;
    std::chrono::steady_clock::time_point last_;
};

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void validate_mel_parameters(int sr, int n_fft, int n_mels, double f_min,
                             double f_max) {
    if (sr <= 0) throw std::invalid_argument("Sample rate must be positive");
    if (!is_power_of_two(n_fft))
        throw std::invalid_argument("n_fft must be a positive power of two");
    if (n_mels <= 0) throw std::invalid_argument("n_mels must be positive");
    if (!std::isfinite(f_min) || !std::isfinite(f_max) || f_min < 0.0 ||
        !(f_min < f_max)) {
        throw std::invalid_argument(
            "Frequency range must be finite and satisfy 0 <= f_min < f_max");
    }
}

std::vector<double> make_mel_points(int n_mels, double f_min, double f_max) {
    const double mel_min = hz_to_mel(f_min);
    const double mel_max = hz_to_mel(f_max);
    const std::size_t point_count = static_cast<std::size_t>(n_mels) + 2;
    std::vector<double> points(point_count);
    for (std::size_t i = 0; i < point_count; ++i) {
        const double mel = mel_min + static_cast<double>(i) *
                                         (mel_max - mel_min) /
                                         (static_cast<double>(n_mels) + 1.0);
        points[i] = mel_to_hz(mel);
        if (!std::isfinite(points[i]) ||
            (i != 0 && !(points[i] > points[i - 1]))) {
            throw std::invalid_argument(
                "Mel points are not finite and strictly increasing");
        }
    }
    return points;
}

struct MelBand {
    std::size_t first;
    std::size_t end;
    std::size_t offset;
};

struct MelWeights {
    std::size_t n_bins;
    std::vector<MelBand> bands;
    std::vector<double> values;
};

MelWeights make_compact_mel_weights(int sr, int n_fft, int n_mels, double f_min,
                                    double f_max) {
    const std::size_t n_bins = static_cast<std::size_t>(n_fft / 2) + 1;
    std::vector<double> fft_freqs(n_bins);
    for (std::size_t k = 0; k < n_bins; ++k) {
        fft_freqs[k] = static_cast<double>(k) * static_cast<double>(sr) /
                       static_cast<double>(n_fft);
    }
    const std::vector<double> points = make_mel_points(n_mels, f_min, f_max);

    MelWeights result{n_bins, std::vector<MelBand>(n_mels), {}};
    std::size_t total_weights = 0;
    for (int m = 0; m < n_mels; ++m) {
        const double f_left = points[static_cast<std::size_t>(m)];
        const double f_right = points[static_cast<std::size_t>(m) + 2];
        const auto first_it =
            std::upper_bound(fft_freqs.begin(), fft_freqs.end(), f_left);
        const auto end_it =
            std::lower_bound(fft_freqs.begin(), fft_freqs.end(), f_right);
        const std::size_t first =
            static_cast<std::size_t>(first_it - fft_freqs.begin());
        const std::size_t end =
            static_cast<std::size_t>(end_it - fft_freqs.begin());
        if (end < first)
            throw std::invalid_argument("Mel filter has an invalid bin range");
        const std::size_t count = end - first;
        if (count > std::numeric_limits<std::size_t>::max() - total_weights)
            throw std::length_error("Mel filter weight count overflows size_t");
        result.bands[static_cast<std::size_t>(m)] =
            MelBand{first, end, total_weights};
        total_weights += count;
    }

    result.values.resize(total_weights);
    for (int m = 0; m < n_mels; ++m) {
        const MelBand& band = result.bands[static_cast<std::size_t>(m)];
        if (band.first == band.end) continue;
        const double f_left = points[static_cast<std::size_t>(m)];
        const double f_center = points[static_cast<std::size_t>(m) + 1];
        const double f_right = points[static_cast<std::size_t>(m) + 2];
        const double enorm = 2.0 / (f_right - f_left);
        for (std::size_t k = band.first; k < band.end; ++k) {
            const double frequency = fft_freqs[k];
            const double rising = (frequency - f_left) / (f_center - f_left);
            const double falling = (f_right - frequency) / (f_right - f_center);
            const double value = std::max(0.0, std::min(rising, falling));
            result.values[band.offset + k - band.first] = value * enorm;
        }
    }
    return result;
}

std::size_t parse_requested_threads() {
    const char* text = std::getenv("MELSPEC_THREADS");
    if (text == nullptr) return std::numeric_limits<std::size_t>::max();
    if (*text == '\0')
        throw std::invalid_argument(
            "MELSPEC_THREADS must be a positive integer");

    std::size_t value = 0;
    for (const char* ch = text; *ch != '\0'; ++ch) {
        if (*ch < '0' || *ch > '9')
            throw std::invalid_argument(
                "MELSPEC_THREADS must be a positive integer");
        const std::size_t digit = static_cast<std::size_t>(*ch - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            throw std::invalid_argument("MELSPEC_THREADS is too large");
        }
        value = value * 10 + digit;
    }
    if (value == 0)
        throw std::invalid_argument(
            "MELSPEC_THREADS must be a positive integer");
    return value;
}

std::size_t usable_cpus() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) return static_cast<std::size_t>(count);
    }
#endif
    return std::max<std::size_t>(
        1, static_cast<std::size_t>(std::thread::hardware_concurrency()));
}

std::size_t worker_count_for(std::size_t groups, std::size_t requested) {
    return std::max<std::size_t>(
        1, std::min(groups, std::min(requested, usable_cpus())));
}

void fft_inplace(std::vector<std::complex<double>>& values,
                 const Twiddles& twiddles) {
    const std::size_t n = values.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n / 2;
        for (; bit != 0 && (j & bit) != 0; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }

    for (std::size_t len = 2; len <= n;) {
        const std::size_t half = len / 2;
        for (std::size_t base = 0; base < n; base += len) {
            for (std::size_t j = 0; j < half; ++j) {
                const std::complex<double> u = values[base + j];
                const std::complex<double> a = values[base + j + half];
                const std::complex<double> w(twiddles.real[half + j],
                                             twiddles.imag[half + j]);
                const std::complex<double> v = a * w;
                values[base + j] = u + v;
                values[base + j + half] = u - v;
            }
        }
        if (len == n) break;
        len *= 2;
    }
}

// Runs body(worker) on `workers` threads (worker 0 on the caller) and rethrows
// the first failure after every started thread has been joined.
template <typename Body>
void run_workers(std::size_t workers, Body&& body) {
    if (workers == 1) {
        body(std::size_t{0});
        return;
    }
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> worker_errors(workers);
    threads.reserve(workers - 1);
    std::exception_ptr launch_error;
    for (std::size_t worker = 1; worker < workers; ++worker) {
        try {
            threads.emplace_back([&, worker] {
                try {
                    body(worker);
                } catch (...) {
                    worker_errors[worker] = std::current_exception();
                }
            });
        } catch (...) {
            launch_error = std::current_exception();
            break;
        }
    }
    std::exception_ptr main_error;
    if (!launch_error) {
        try {
            body(std::size_t{0});
        } catch (...) {
            main_error = std::current_exception();
        }
    }
    for (std::thread& thread : threads) thread.join();

    if (launch_error) std::rethrow_exception(launch_error);
    if (main_error) std::rethrow_exception(main_error);
    for (const std::exception_ptr& error : worker_errors) {
        if (error) std::rethrow_exception(error);
    }
}

#if MELSPEC_CUDA
// Waits until ready() holds or stop is set; returns ready(). Sleeps after a
// short spin so idle host threads leave the cores to those doing work.
template <typename Ready>
bool wait_for(Ready&& ready, const std::atomic<bool>& stop) {
    for (int spins = 0;; ++spins) {
        if (ready()) return true;
        if (stop.load(std::memory_order_acquire)) return ready();
        if (spins < 64)
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
}

// GPU path: the device computes float Mel rows chunk by chunk (see
// melspec_cuda.cu) while host threads widen each chunk into the double
// output. While the CUDA start-up thread is still creating the context, the
// host threads convert the samples to float (exact: they are pcm / 32768),
// one of them sizes (zero-fills) the output, and the calling thread faults
// in write_output's float copy, as in compute_with_mkl. Returns false, with
// the output possibly already sized, when no GPU is usable.
template <typename PrepareOutput>
bool compute_with_cuda(const double* samples, std::size_t n_samples,
                       std::size_t n_frames, std::size_t fft_size,
                       std::size_t hop, const MelWeights& mel,
                       std::size_t requested_threads,
                       PrepareOutput&& prepare_output, bool trace) {
    using Clock = std::chrono::steady_clock;
    const Clock::time_point start = Clock::now();
    const auto seconds = [&] {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    const std::size_t mel_count = mel.bands.size();
    std::vector<std::size_t> first(mel_count), end(mel_count),
        offset(mel_count);
    for (std::size_t m = 0; m < mel_count; ++m) {
        first[m] = mel.bands[m].first;
        end[m] = mel.bands[m].end;
        offset[m] = mel.bands[m].offset;
    }
    // Every element is written by the conversion below.
    std::unique_ptr<float[]> converted(new float[n_samples]);
    const melspec_cuda::Job job{converted.get(), n_samples,    n_frames,
                                fft_size,        hop,          mel_count,
                                first.data(),    end.data(),   offset.data(),
                                mel.values.data(), mel.values.size()};

    constexpr std::size_t kConvertPiece = std::size_t(1) << 20;
    const std::size_t pieces = (n_samples + kConvertPiece - 1) / kConvertPiece;
    std::atomic<std::size_t> next_piece{0};
    std::atomic<std::size_t> pieces_done{0};
    const auto convert = [&] {
        for (;;) {
            const std::size_t piece =
                next_piece.fetch_add(1, std::memory_order_relaxed);
            if (piece >= pieces) return;
            const std::size_t lo = piece * kConvertPiece;
            const std::size_t hi = std::min(n_samples, lo + kConvertPiece);
            for (std::size_t i = lo; i < hi; ++i)
                converted[i] = static_cast<float>(samples[i]);
            pieces_done.fetch_add(1, std::memory_order_release);
        }
    };

    enum : int { kPending, kReady, kFailed };
    std::atomic<int> output_state{kPending};
    double* output = nullptr;  // published by output_state (release/acquire)
    // Chunk c is handed out through slot c % kSlots (the GPU side cycles its
    // pinned staging buffers the same way); rows of all chunks are claimed
    // as consecutive tickets.
    constexpr std::size_t kSlots = melspec_cuda::kSlots;
    struct Chunk {
        std::size_t frame0;
        std::size_t count;
        const float* rows;
    };
    Chunk chunks[kSlots] = {};
    std::atomic<std::size_t> published{0};
    std::atomic<std::size_t> next_ticket{0};
    std::atomic<std::size_t> rows_done[kSlots] = {};
    std::atomic<bool> stop{false};

    const auto helper = [&](std::size_t index) {
        if (index == 0) {
            try {
                output = prepare_output();
            } catch (...) {
                output_state.store(kFailed, std::memory_order_release);
                throw;
            }
            output_state.store(kReady, std::memory_order_release);
        }
        convert();
        if (!wait_for(
                [&] {
                    return output_state.load(std::memory_order_acquire) !=
                           kPending;
                },
                stop) ||
            output_state.load(std::memory_order_acquire) != kReady)
            return;
        for (;;) {
            const std::size_t ticket =
                next_ticket.fetch_add(1, std::memory_order_relaxed);
            const std::size_t c = ticket / mel_count;
            if (!wait_for(
                    [&] {
                        return published.load(std::memory_order_acquire) > c;
                    },
                    stop))
                return;
            const Chunk& chunk = chunks[c % kSlots];
            const std::size_t m = ticket % mel_count;
            const float* source = chunk.rows + m * chunk.count;
            double* destination = output + m * n_frames + chunk.frame0;
            for (std::size_t f = 0; f < chunk.count; ++f)
                destination[f] = source[f];
            rows_done[c % kSlots].fetch_add(1, std::memory_order_release);
        }
    };

    const std::size_t helpers = std::max<std::size_t>(
        1, std::min(requested_threads, usable_cpus()) - 1);
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(helpers);
    std::exception_ptr main_error;
    bool gpu = false;
    try {
        threads.reserve(helpers);
        for (std::size_t i = 0; i < helpers; ++i)
            threads.emplace_back([&, i] {
                try {
                    helper(i);
                } catch (...) {
                    errors[i] = std::current_exception();
                }
            });
#if MELSPEC_HAS_RESERVE
        if (melspec_reserve_allocation != nullptr)
            melspec_reserve_allocation(mel_count * n_frames * sizeof(float));
#endif
        convert();
        const double converted_at = seconds();
        gpu = melspec_cuda::available();
        const double ready_at = seconds();
        if (gpu) {
            wait_for(
                [&] {
                    return pieces_done.load(std::memory_order_acquire) ==
                           pieces;
                },
                stop);
            if (trace)
                std::fprintf(stderr,
                             "trace cuda host: converted %.4f gpu ready "
                             "%.4f samples ready %.4f s\n",
                             converted_at, ready_at, seconds());
            std::size_t consumed = 0;
            melspec_cuda::compute(
                job,
                [&](std::size_t frame0, std::size_t count,
                    const float* rows) {
                    const std::size_t slot = consumed % kSlots;
                    chunks[slot] = Chunk{frame0, count, rows};
                    rows_done[slot].store(0, std::memory_order_relaxed);
                    published.store(++consumed, std::memory_order_release);
                    while (rows_done[slot].load(std::memory_order_acquire) <
                           mel_count) {
                        if (output_state.load(std::memory_order_acquire) ==
                            kFailed)
                            throw std::runtime_error(
                                "output preparation failed");
                        std::this_thread::yield();
                    }
                },
                trace);
        }
    } catch (...) {
        main_error = std::current_exception();
    }
    stop.store(true, std::memory_order_release);
    for (std::thread& thread : threads) thread.join();
    if (!gpu && !main_error) return false;
    // A failed output preparation is the root cause of a consume failure.
    for (const std::exception_ptr& error : errors)
        if (error) std::rethrow_exception(error);
    if (main_error) std::rethrow_exception(main_error);
    return true;
}
#endif

#if MELSPEC_MKL
#ifndef MELSPEC_MKL_BATCH
#define MELSPEC_MKL_BATCH 1
#endif
#ifndef MELSPEC_FFT_FLOAT
#define MELSPEC_FFT_FLOAT 0
#endif
#ifndef MELSPEC_FREQ_WINDOW
#define MELSPEC_FREQ_WINDOW 1
#endif

template <typename Real>
class DftiPlan {
   public:
    // `batch` real transforms of length n, stored back to back with a stride
    // of n + 2 doubles so each fits its n/2+1 complex outputs in place; or,
    // with out_of_place, one transform from caller input to a separate
    // n/2+1 complex output.
    DftiPlan(std::size_t n, std::size_t batch, bool out_of_place = false) {
        // DftiCreateDescriptor is a macro, so the precision is computed
        // first: a template argument list would split its arguments.
        const DFTI_CONFIG_VALUE precision =
            std::is_same<Real, float>::value ? DFTI_SINGLE : DFTI_DOUBLE;
        MKL_LONG status = DftiCreateDescriptor(&handle_, precision, DFTI_REAL,
                                               1, static_cast<MKL_LONG>(n));
        if (status == 0)
            status = DftiSetValue(handle_, DFTI_CONJUGATE_EVEN_STORAGE,
                                  DFTI_COMPLEX_COMPLEX);
        if (status == 0) status = DftiSetValue(handle_, DFTI_THREAD_LIMIT, 1);
        if (status == 0 && out_of_place)
            status = DftiSetValue(handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        if (status == 0 && batch > 1) {
            status = DftiSetValue(handle_, DFTI_NUMBER_OF_TRANSFORMS,
                                  static_cast<MKL_LONG>(batch));
            if (status == 0)
                status = DftiSetValue(handle_, DFTI_INPUT_DISTANCE,
                                      static_cast<MKL_LONG>(n + 2));
            if (status == 0)
                status = DftiSetValue(handle_, DFTI_OUTPUT_DISTANCE,
                                      static_cast<MKL_LONG>(n / 2 + 1));
        }
        if (status == 0) status = DftiCommitDescriptor(handle_);
        if (status != 0) {
            if (handle_ != nullptr) DftiFreeDescriptor(&handle_);
            throw std::runtime_error("MKL DFTI descriptor setup failed");
        }
    }
    ~DftiPlan() {
        DftiFreeDescriptor(&handle_);
    }
    DftiPlan(const DftiPlan&) = delete;
    DftiPlan& operator=(const DftiPlan&) = delete;

    void forward(Real* data) const {
        if (DftiComputeForward(handle_, data) != 0)
            throw std::runtime_error("MKL DFTI forward transform failed");
    }
    // Out-of-place plans only; MKL leaves the input of such a transform
    // unmodified, but its C interface takes a non-const pointer.
    void forward(const Real* input, Real* output) const {
        if (DftiComputeForward(handle_, const_cast<Real*>(input), output) != 0)
            throw std::runtime_error("MKL DFTI forward transform failed");
    }

   private:
    DFTI_DESCRIPTOR_HANDLE handle_ = nullptr;
};

// Frames per output tile: Mel results of a tile are gathered as
// [mel][frame] so each Mel row is written to the output contiguously.
constexpr std::size_t kTileFrames = 16;
constexpr std::size_t kFftBatch = MELSPEC_MKL_BATCH;
static_assert(kTileFrames % kFftBatch == 0, "batch must divide the tile");
// Frames whose Mel sums are accumulated together (independent FMA chains).
constexpr std::size_t kMelGroup = 4;  // 8 measured no faster
static_assert(kTileFrames % kMelGroup == 0, "groups must tile a tile");
// Tiles handed out per scheduling step.
constexpr std::size_t kTilesPerClaim = 2;

// The periodic Hann window is w[n] = 1/2 - (e^{2 pi i n/N} + e^{-2 pi i
// n/N})/4, so the DFT of a windowed frame is exactly
//     Y[k] = X[k]/2 - (X[k-1] + X[k+1])/4
// with X the DFT of the unwindowed frame. Transforming straight from the
// input samples (out of place) and applying the window here, only on the
// bins the Mel bands read, removes the per-frame windowed copy.
// Float FFTs read a float copy of each tile's samples (see compute_with_mkl).
template <typename Real>
constexpr bool kFreqWindow = MELSPEC_FREQ_WINDOW && kFftBatch == 1;

// Writes |Y[k]|^2 for the active bins of the frame at `frame` into power.
template <typename Real, typename Lap>
void frequency_window_power(const DftiPlan<Real>& plan, const Real* frame,
                            std::vector<Real>& spectrum, std::size_t n_bins,
                            std::size_t active_first, Real* power,
                            std::size_t count, Lap& lap) {
    lap(0);
    Real* x = spectrum.data() + 2;  // x[2k], x[2k+1]: Re, Im of X[k]
    plan.forward(frame, x);
    lap(1);
    // X is conjugate-even for real input: X[-1] = conj(X[1]) and
    // X[n/2+1] = conj(X[n/2-1]).
    const std::size_t half = n_bins - 1;
    x[-2] = x[2];
    x[-1] = -x[3];
    x[2 * (half + 1)] = x[2 * (half - 1)];
    x[2 * (half + 1) + 1] = -x[2 * (half - 1) + 1];
    const Real* bin = x + 2 * active_first;
    for (std::size_t k = 0; k < count; ++k) {
        const Real re = Real(0.5) * bin[2 * k] -
                        Real(0.25) * (bin[2 * k - 2] + bin[2 * k + 2]);
        const Real im = Real(0.5) * bin[2 * k + 1] -
                        Real(0.25) * (bin[2 * k - 1] + bin[2 * k + 3]);
        power[k] = re * re + im * im;
    }
    lap(2);
}

// Real is the FFT working precision; the Mel accumulation is always double.
//
// prepare_output() sizes the (zero-initialized) output and returns its data.
// Worker 1 (worker 0 if alone) runs it while the others already compute tiles,
// which they keep in a private stash until the output is ready, so the
// single-threaded zero fill of the output overlaps with the FFT work.
template <typename Real, typename PrepareOutput>
void compute_with_mkl(const double* samples, std::size_t n_frames,
                      std::size_t fft_size, std::size_t hop,
                      const MelWeights& mel, const std::vector<double>& window,
                      std::size_t requested_threads,
                      PrepareOutput&& prepare_output, bool trace) {
    const std::size_t n_bins = fft_size / 2 + 1;
    std::size_t active_first = n_bins;
    std::size_t active_end = 0;
    for (const MelBand& band : mel.bands) {
        if (band.first == band.end) continue;
        active_first = std::min(active_first, band.first);
        active_end = std::max(active_end, band.end);
    }
    const std::size_t mel_count = mel.bands.size();
    if (active_end <= active_first) {
        prepare_output();  // every Mel band is empty: all zeros
        return;
    }
    const std::size_t active_bins = active_end - active_first;
    const std::size_t stride = fft_size + 2;

    const std::size_t tiles =
        n_frames / kTileFrames + (n_frames % kTileFrames != 0 ? 1 : 0);
    const std::size_t claims =
        tiles / kTilesPerClaim + (tiles % kTilesPerClaim != 0 ? 1 : 0);
    const std::size_t workers = worker_count_for(claims, requested_threads);
    std::atomic<std::size_t> next_claim{0};

    using Clock = std::chrono::steady_clock;
    const Clock::time_point launch = Clock::now();
    std::vector<double> worker_times(trace ? 3 * workers : 0);
    // Per-worker seconds in each traced stage (see `lap`).
    constexpr int kStages = 6;
    std::vector<double> stage_times(trace ? kStages * workers : 0);

    enum : int { kPending, kReady, kFailed };
    std::atomic<int> output_state{kPending};
    double* output = nullptr;  // published by output_state (release/acquire)

    // Two serial set-up jobs overlap the FFT work: one worker sizes (and so
    // zero-fills) the output while the others compute into private stashes,
    // and worker 0 -- the calling thread -- faults in the float copy that
    // main.cpp's write_output allocates next. That copy has to be first
    // touched by the calling thread: the conversion loop there writes all of
    // it, and remote NUMA writes cost more than the saved faults. Doing the
    // two jobs on different workers keeps either from outlasting the
    // shortest kernels.
    const std::size_t preparer = workers > 1 ? 1 : 0;
    const auto reserve_float_copy = [&] {
#if MELSPEC_HAS_RESERVE
        if (melspec_reserve_allocation != nullptr)
            melspec_reserve_allocation(mel_count * n_frames * sizeof(float));
#endif
    };
    run_workers(workers, [&](std::size_t worker) {
        // The output is twice the float copy's size, so its allocation
        // never takes the reserved block.
        if (worker == 0 && preparer != 0) reserve_float_copy();
        if (worker == preparer) {
            try {
                output = prepare_output();
            } catch (...) {
                output_state.store(kFailed, std::memory_order_release);
                throw;
            }
            output_state.store(kReady, std::memory_order_release);
            if (preparer == 0) reserve_float_copy();
        }
        bool ready = worker == preparer;
        std::vector<std::pair<std::size_t, std::vector<double>>> stash;
        std::size_t stashed_tiles = 0;
        const auto store_tile = [&](std::size_t tile, const double* values) {
            const std::size_t frame0 = tile * kTileFrames;
            const std::size_t count = std::min(kTileFrames, n_frames - frame0);
            for (std::size_t m = 0; m < mel_count; ++m) {
                const double* row = values + m * kTileFrames;
                double* destination = output + m * n_frames + frame0;
                for (std::size_t f = 0; f < count; ++f) destination[f] = row[f];
            }
        };
        // Returns false if the output could not be prepared.
        const auto become_ready = [&](bool wait) {
            int state = output_state.load(std::memory_order_acquire);
            while (wait && state == kPending) {
                std::this_thread::yield();
                state = output_state.load(std::memory_order_acquire);
            }
            if (state == kReady) {
                ready = true;
                for (const auto& entry : stash)
                    store_tile(entry.first, entry.second.data());
                stash.clear();
            }
            return state != kFailed;
        };

        std::vector<Real> window_real(window.begin(), window.end());
        const DftiPlan<Real> plan(fft_size, kFftBatch, kFreqWindow<Real>);
        const Clock::time_point planned = Clock::now();
        std::vector<Real> buffer(kFreqWindow<Real> ? 0 : stride * kFftBatch);
        // Unwindowed spectrum X[-1..n/2+1] for the frequency-domain window;
        // bin k is at spectrum[2 * (k + 1)].
        std::vector<Real> spectrum(kFreqWindow<Real> ? 2 * (n_bins + 2) : 0);
        // A float FFT cannot read the double samples directly: each tile's
        // sample span is converted once here (about 1/13 of converting every
        // frame separately).
        constexpr bool kConvertSamples =
            kFreqWindow<Real> && !std::is_same<Real, double>::value;
        std::vector<Real> converted(
            kConvertSamples ? (kTileFrames - 1) * hop + fft_size : 0);
        std::vector<Real> power(active_bins);
        // Power rows of a group of frames for the frequency-window path.
        std::vector<Real> power_group(
            kFreqWindow<Real> ? kMelGroup * active_bins : 0);
        std::vector<double> tile_out(mel_count * kTileFrames);

        // window, fft, power, mel, tile store or stash, claims and flushes
        double stage[kStages] = {0, 0, 0, 0, 0, 0};
        Clock::time_point mark = Clock::now();
        const auto lap = [&](int index) {
            if (!trace) return;
            const Clock::time_point now = Clock::now();
            stage[index] += std::chrono::duration<double>(now - mark).count();
            mark = now;
        };
        // Mel bands of one frame's power (bins from active_first) into
        // column f of the tile.
        const auto mel_frame = [&](const std::vector<Real>& frame_power,
                                   std::size_t f) {
            for (std::size_t m = 0; m < mel_count; ++m) {
                const MelBand& band = mel.bands[m];
                const double* weights = mel.values.data() + band.offset;
                const Real* p =
                    frame_power.data() + (band.first - active_first);
                const std::size_t length = band.end - band.first;
                double sum = 0.0;
                for (std::size_t k = 0; k < length; ++k)
                    sum += p[k] * weights[k];
                tile_out[m * kTileFrames + f] = sum;
            }
        };

        // Mel bands of `group` (1..kMelGroup) frames whose power rows start
        // at rows, into tile columns f0... Every frame keeps its own
        // ascending-bin sum, so results equal mel_frame's bit for bit; the
        // independent sums only let the FMA chains overlap instead of each
        // waiting on the previous add.
        const auto mel_group = [&](const Real* rows, std::size_t f0,
                                   std::size_t group) {
            const Real* row[kMelGroup];
            for (std::size_t g = 0; g < kMelGroup; ++g)
                row[g] = rows + std::min(g, group - 1) * active_bins;
            for (std::size_t m = 0; m < mel_count; ++m) {
                const MelBand& band = mel.bands[m];
                const double* weights = mel.values.data() + band.offset;
                const std::size_t offset = band.first - active_first;
                const std::size_t length = band.end - band.first;
                double sums[kMelGroup] = {};
                for (std::size_t k = 0; k < length; ++k) {
                    const double w = weights[k];
                    for (std::size_t g = 0; g < kMelGroup; ++g)
                        sums[g] += row[g][offset + k] * w;
                }
                double* out = tile_out.data() + m * kTileFrames + f0;
                for (std::size_t g = 0; g < group; ++g) out[g] = sums[g];
            }
        };

        for (;;) {
            if (!ready && !become_ready(false)) return;
            lap(4);  // flushing the stash is storing tiles
            const std::size_t claim =
                next_claim.fetch_add(1, std::memory_order_relaxed);
            lap(5);
            if (claim >= claims) break;
            const std::size_t tile_end =
                std::min(tiles, (claim + 1) * kTilesPerClaim);
            for (std::size_t tile = claim * kTilesPerClaim; tile < tile_end;
                 ++tile) {
                const std::size_t frame0 = tile * kTileFrames;
                const std::size_t count =
                    std::min(kTileFrames, n_frames - frame0);
                if constexpr (kFreqWindow<Real>) {
                    const Real* tile_samples;
                    if constexpr (kConvertSamples) {
                        const double* source = samples + frame0 * hop;
                        const std::size_t span = (count - 1) * hop + fft_size;
                        for (std::size_t i = 0; i < span; ++i)
                            converted[i] = static_cast<Real>(source[i]);
                        tile_samples = converted.data();
                    } else {
                        tile_samples = samples + frame0 * hop;
                    }
                    for (std::size_t f0 = 0; f0 < count; f0 += kMelGroup) {
                        const std::size_t group =
                            std::min(kMelGroup, count - f0);
                        for (std::size_t g = 0; g < group; ++g)
                            frequency_window_power(
                                plan, tile_samples + (f0 + g) * hop, spectrum,
                                n_bins, active_first,
                                power_group.data() + g * active_bins,
                                active_bins, lap);
                        mel_group(power_group.data(), f0, group);
                        lap(3);
                    }
                } else {
                    for (std::size_t batch0 = 0; batch0 < count;
                         batch0 += kFftBatch) {
                        // Frames past the end of a partial batch repeat the
                        // last valid frame; their results are never stored.
                        for (std::size_t b = 0; b < kFftBatch; ++b) {
                            const std::size_t f =
                                std::min(batch0 + b, count - 1);
                            const double* src = samples + (frame0 + f) * hop;
                            Real* data = buffer.data() + b * stride;
                            for (std::size_t i = 0; i < fft_size; ++i)
                                data[i] =
                                    static_cast<Real>(src[i]) * window_real[i];
                        }
                        lap(0);
                        plan.forward(buffer.data());
                        lap(1);
                        const std::size_t batch_count =
                            std::min(kFftBatch, count - batch0);
                        for (std::size_t b = 0; b < batch_count; ++b) {
                            const Real* bin =
                                buffer.data() + b * stride + 2 * active_first;
                            for (std::size_t k = 0; k < active_bins; ++k) {
                                const Real re = bin[2 * k];
                                const Real im = bin[2 * k + 1];
                                power[k] = re * re + im * im;
                            }
                            lap(2);
                            mel_frame(power, batch0 + b);
                            lap(3);
                        }
                    }
                }
                if (ready) {
                    store_tile(tile, tile_out.data());
                } else {
                    stash.emplace_back(tile, tile_out);
                    ++stashed_tiles;
                }
                lap(4);
            }
        }
        if (!ready && !become_ready(true)) return;
        lap(5);
        if (trace) {
            for (int i = 0; i < kStages; ++i)
                stage_times[kStages * worker + i] = stage[i];
            worker_times[3 * worker] =
                std::chrono::duration<double>(planned - launch).count();
            worker_times[3 * worker + 1] =
                std::chrono::duration<double>(Clock::now() - launch).count();
            worker_times[3 * worker + 2] = static_cast<double>(stashed_tiles);
        }
    });
    if (trace) {
        double lo[3] = {1e9, 1e9, 1e9}, hi[3] = {0, 0, 0};
        for (std::size_t w = 0; w < workers; ++w)
            for (int i = 0; i < 3; ++i) {
                lo[i] = std::min(lo[i], worker_times[3 * w + i]);
                hi[i] = std::max(hi[i], worker_times[3 * w + i]);
            }
        std::fprintf(stderr,
                     "trace workers %zu planned %.4f..%.4f done %.4f..%.4f "
                     "stashed %.0f..%.0f\n",
                     workers, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        double planned_sum = 0, done_sum = 0;
        for (std::size_t w = 0; w < workers; ++w) {
            planned_sum += worker_times[3 * w];
            done_sum += worker_times[3 * w + 1];
        }
        std::fprintf(stderr, "trace mean planned %.4f done %.4f\n",
                     planned_sum / static_cast<double>(workers),
                     done_sum / static_cast<double>(workers));
        double total[kStages] = {0, 0, 0, 0, 0, 0};
        for (std::size_t w = 0; w < workers; ++w)
            for (int i = 0; i < kStages; ++i)
                total[i] += stage_times[kStages * w + i];
        // Summed over workers, as seconds per worker on average.
        const double n = static_cast<double>(workers);
        std::fprintf(stderr,
                     "trace avg window %.4f fft %.4f power %.4f mel %.4f "
                     "store %.4f sched %.4f\n",
                     total[0] / n, total[1] / n, total[2] / n, total[3] / n,
                     total[4] / n, total[5] / n);
    }
}
#endif

}  // namespace

// HTK mel scale conversions.
double hz_to_mel(double f) {
    return 2595.0 * std::log10(1.0 + f / 700.0);
}

double mel_to_hz(double m) {
    return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0);
}

// Public complex FFT helper retained for callers that need the complete real
// FFT.
void compute_rfft(const std::vector<double>& frame,
                  std::vector<std::complex<double>>& out) {
    if (frame.empty() ||
        frame.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !is_power_of_two(static_cast<int>(frame.size()))) {
        throw std::invalid_argument(
            "RFFT input length must be a positive power of two");
    }
    const std::size_t n = frame.size();
    std::vector<std::complex<double>> values(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = {frame[i], 0.0};
    if (n > 1) fft_inplace(values, make_twiddles(n));
    out.resize(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) out[k] = values[k];
}

// Build the dense public filterbank representation.
std::vector<std::vector<double>> build_mel_filterbank(int sr, int n_fft,
                                                      int n_mels, double f_min,
                                                      double f_max) {
    validate_mel_parameters(sr, n_fft, n_mels, f_min, f_max);
    const MelWeights compact =
        make_compact_mel_weights(sr, n_fft, n_mels, f_min, f_max);
    std::vector<std::vector<double>> dense(
        static_cast<std::size_t>(n_mels),
        std::vector<double>(compact.n_bins, 0.0));
    for (int m = 0; m < n_mels; ++m) {
        const MelBand& band = compact.bands[static_cast<std::size_t>(m)];
        for (std::size_t k = band.first; k < band.end; ++k) {
            dense[static_cast<std::size_t>(m)][k] =
                compact.values[band.offset + k - band.first];
        }
    }
    return dense;
}

void compute_melspectrogram(const std::vector<double>& y, int sr, int n_fft,
                            int hop_length, int n_mels, double f_min,
                            double f_max, std::vector<double>& output) {
    StageTrace trace;
    validate_mel_parameters(sr, n_fft, n_mels, f_min, f_max);
    if (hop_length <= 0)
        throw std::invalid_argument("hop_length must be positive");
    if (y.size() < static_cast<std::size_t>(n_fft))
        throw std::runtime_error("Audio too short for given n_fft");

    const std::size_t n_samples = y.size();
    const std::size_t fft_size = static_cast<std::size_t>(n_fft);
    const std::size_t hop = static_cast<std::size_t>(hop_length);
    const std::size_t n_frames = 1 + (n_samples - fft_size) / hop;
    const std::size_t mel_count = static_cast<std::size_t>(n_mels);
    if (n_frames > std::numeric_limits<std::size_t>::max() / mel_count)
        throw std::length_error("Output element count overflows size_t");
    const std::size_t output_count = mel_count * n_frames;

    const std::size_t requested_threads = parse_requested_threads();
    const MelWeights mel =
        make_compact_mel_weights(sr, n_fft, n_mels, f_min, f_max);

    // Preserve the input before resize if callers intentionally reuse one
    // vector for both input and output. The ordinary out-of-place path does
    // not copy the signal.
    std::vector<double> alias_input;
    const std::vector<double>* input = &y;
    if (&y == &output) {
        alias_input = y;
        input = &alias_input;
    }
    const double* input_samples = input->data();

    std::vector<double> window(fft_size);
    for (std::size_t i = 0; i < fft_size; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                          static_cast<double>(fft_size)));
    }
    trace.mark("setup");

#if MELSPEC_CUDA
    // The start-up thread must finish before main returns: exiting while it
    // is inside the CUDA runtime races with the runtime's own teardown.
    // compute_with_cuda waits for it; so does the n_fft == 1 path here.
    if (n_fft <= 1) melspec_cuda::available();
    if (n_fft > 1 &&
        compute_with_cuda(
            input_samples, input->size(), n_frames, fft_size, hop, mel,
            requested_threads,
            [&] {
                output.resize(output_count);
                return output.data();
            },
            trace.enabled())) {
        trace.mark("kernel");
        return;
    }
#endif

#if MELSPEC_MKL
    if (n_fft > 1) {
        using FftReal =
            std::conditional<MELSPEC_FFT_FLOAT != 0, float, double>::type;
        compute_with_mkl<FftReal>(
            input_samples, n_frames, fft_size, hop, mel, window,
            requested_threads,
            [&] {
                output.resize(output_count);
                return output.data();
            },
            trace.enabled());
        trace.mark("kernel");
        return;
    }
#endif

    // The periodic Hann window of length one is exactly zero.
    if (n_fft == 1) {
        output.resize(output_count);
        std::fill(output.begin(), output.end(), 0.0);
        return;
    }

    const std::size_t half_fft = fft_size / 2;
    const std::size_t n_bins = half_fft + 1;
    const Twiddles twiddles = make_twiddles(half_fft);
    std::vector<double> rfft_twiddle_real(n_bins);
    std::vector<double> rfft_twiddle_imag(n_bins);
    for (std::size_t k = 0; k < n_bins; ++k) {
        const double angle =
            -2.0 * kPi * static_cast<double>(k) / static_cast<double>(fft_size);
        rfft_twiddle_real[k] = std::cos(angle);
        rfft_twiddle_imag[k] = std::sin(angle);
    }

    std::vector<std::size_t> bit_reverse(half_fft);
    for (std::size_t i = 0, reversed = 0; i < half_fft; ++i) {
        bit_reverse[i] = reversed;
        std::size_t bit = half_fft / 2;
        for (; bit != 0 && (reversed & bit) != 0; bit >>= 1) {
            reversed ^= bit;
        }
        reversed ^= bit;
    }

    std::size_t active_first = n_bins;
    std::size_t active_end = 0;
    for (const MelBand& band : mel.bands) {
        if (band.first == band.end) continue;
        active_first = std::min(active_first, band.first);
        active_end = std::max(active_end, band.end);
    }
    const bool low_power_only = active_end <= half_fft / 2 + 1;

    const std::size_t group_count =
        n_frames / kFrameLanes + (n_frames % kFrameLanes != 0 ? 1 : 0);
    const std::size_t workers =
        worker_count_for(group_count, requested_threads);

    const std::size_t scratch_count = half_fft * 2 + n_bins;
    // Lane groups per scheduling step, and the frames they cover.
    constexpr std::size_t kGroupsPerClaim = 8;
    constexpr std::size_t kClaimFrames = kGroupsPerClaim * kFrameLanes;
    // Computes the frames of groups [first_group, end_group) into `tile`,
    // laid out [mel][frame - first frame] with a row stride of kClaimFrames.
    auto process_block = [&](std::size_t first_group, std::size_t end_group,
                             std::vector<FrameVec>& scratch,
                             std::vector<double>& tile) {
        const std::size_t begin =
            first_group *
            kFrameLanes;  // first_group is strictly before the last group
        const std::size_t end =
            end_group == group_count ? n_frames : end_group * kFrameLanes;

        FrameVec* real = scratch.data();
        FrameVec* imag = real + half_fft;
        FrameVec* power = imag + half_fft;
        double input_lanes[kFrameLanes];

        for (std::size_t frame = begin; frame < end;) {
            const std::size_t count = std::min(kFrameLanes, end - frame);
            const std::size_t last_frame = frame + count - 1;
            for (std::size_t i = 0; i < half_fft; ++i) {
                const std::size_t reversed = bit_reverse[i];
                for (std::size_t lane = 0; lane < kFrameLanes; ++lane) {
                    const std::size_t current_frame =
                        lane < count ? frame + lane : last_frame;
                    const std::size_t sample = current_frame * hop;
                    input_lanes[lane] =
                        input_samples[sample + 2 * i] * window[2 * i];
                }
                real[reversed] = make_frame_vec(input_lanes);
                for (std::size_t lane = 0; lane < kFrameLanes; ++lane) {
                    const std::size_t current_frame =
                        lane < count ? frame + lane : last_frame;
                    const std::size_t sample = current_frame * hop;
                    input_lanes[lane] =
                        input_samples[sample + 2 * i + 1] * window[2 * i + 1];
                }
                imag[reversed] = make_frame_vec(input_lanes);
            }

#if MELSPEC_FUSED_FFT
            std::size_t len = 2;
            for (; len <= half_fft / 2; len *= 4) {
                const std::size_t q = len / 2;
                const std::size_t two_len = 2 * len;
                for (std::size_t base = 0; base < half_fft; base += two_len) {
                    for (std::size_t j = 0; j < q; ++j) {
                        const std::size_t a = base + j;
                        const std::size_t b = a + q;
                        const std::size_t c = a + len;
                        const std::size_t d = c + q;

                        const FrameVec a_real = real[a];
                        const FrameVec a_imag = imag[a];
                        const FrameVec b_real = real[b];
                        const FrameVec b_imag = imag[b];
                        const FrameVec c_real = real[c];
                        const FrameVec c_imag = imag[c];
                        const FrameVec d_real = real[d];
                        const FrameVec d_imag = imag[d];

                        const std::size_t first_twiddle = q + j;
                        const double first_wr = twiddles.real[first_twiddle];
                        const double first_wi = twiddles.imag[first_twiddle];
                        const FrameVec t_real =
                            b_real * first_wr - b_imag * first_wi;
                        const FrameVec t_imag =
                            b_real * first_wi + b_imag * first_wr;
                        const FrameVec u_real =
                            d_real * first_wr - d_imag * first_wi;
                        const FrameVec u_imag =
                            d_real * first_wi + d_imag * first_wr;

                        const FrameVec upper_a_real = a_real + t_real;
                        const FrameVec upper_a_imag = a_imag + t_imag;
                        const FrameVec upper_b_real = a_real - t_real;
                        const FrameVec upper_b_imag = a_imag - t_imag;
                        const FrameVec upper_c_real = c_real + u_real;
                        const FrameVec upper_c_imag = c_imag + u_imag;
                        const FrameVec upper_d_real = c_real - u_real;
                        const FrameVec upper_d_imag = c_imag - u_imag;

                        const std::size_t second_twiddle_a = len + j;
                        const std::size_t second_twiddle_b = len + j + q;
                        const double second_a_wr =
                            twiddles.real[second_twiddle_a];
                        const double second_a_wi =
                            twiddles.imag[second_twiddle_a];
                        const double second_b_wr =
                            twiddles.real[second_twiddle_b];
                        const double second_b_wi =
                            twiddles.imag[second_twiddle_b];
                        const FrameVec v_real = upper_c_real * second_a_wr -
                                                upper_c_imag * second_a_wi;
                        const FrameVec v_imag = upper_c_real * second_a_wi +
                                                upper_c_imag * second_a_wr;
                        const FrameVec z_real = upper_d_real * second_b_wr -
                                                upper_d_imag * second_b_wi;
                        const FrameVec z_imag = upper_d_real * second_b_wi +
                                                upper_d_imag * second_b_wr;

                        real[a] = upper_a_real + v_real;
                        imag[a] = upper_a_imag + v_imag;
                        real[b] = upper_b_real + z_real;
                        imag[b] = upper_b_imag + z_imag;
                        real[c] = upper_a_real - v_real;
                        imag[c] = upper_a_imag - v_imag;
                        real[d] = upper_b_real - z_real;
                        imag[d] = upper_b_imag - z_imag;
                    }
                }
            }
#else
            std::size_t len = 2;
#endif
            for (; len <= half_fft;) {
                const std::size_t stage_half = len / 2;
                for (std::size_t base = 0; base < half_fft; base += len) {
                    for (std::size_t j = 0; j < stage_half; ++j) {
                        const std::size_t left = base + j;
                        const std::size_t right = left + stage_half;
                        const double wr = twiddles.real[stage_half + j];
                        const double wi = twiddles.imag[stage_half + j];
                        const FrameVec ar = real[right];
                        const FrameVec ai = imag[right];
                        const FrameVec vr = ar * wr - ai * wi;
                        const FrameVec vi = ar * wi + ai * wr;
                        const FrameVec ur = real[left];
                        const FrameVec ui = imag[left];
                        real[left] = ur + vr;
                        imag[left] = ui + vi;
                        real[right] = ur - vr;
                        imag[right] = ui - vi;
                    }
                }
                if (len == half_fft) break;
                len *= 2;
            }

            if (low_power_only) {
                if (active_first == 0 && active_end > 0) {
                    const FrameVec dc = real[0] + imag[0];
                    power[0] = dc * dc;
                }
                if ((half_fft & 1u) == 0) {
                    const std::size_t center = half_fft / 2;
                    if (active_first <= center && center < active_end) {
                        power[center] = real[center] * real[center] +
                                        imag[center] * imag[center];
                    }
                }

                const std::size_t first =
                    std::max<std::size_t>(active_first, 1);
                const std::size_t end =
                    std::min(active_end, (half_fft + 1) / 2);
                for (std::size_t k = first; k < end; ++k) {
                    const std::size_t partner = half_fft - k;
                    const FrameVec left_real = real[k];
                    const FrameVec left_imag = imag[k];
                    const FrameVec right_real = real[partner];
                    const FrameVec right_imag = imag[partner];
                    const FrameVec sum_real = left_real + right_real;
                    const FrameVec sum_imag = left_imag - right_imag;
                    const FrameVec diff_real = left_real - right_real;
                    const FrameVec diff_imag = left_imag + right_imag;
                    const double wr = rfft_twiddle_real[k];
                    const double wi = rfft_twiddle_imag[k];
                    const FrameVec correction_real =
                        wr * diff_imag + wi * diff_real;
                    const FrameVec correction_imag =
                        -wr * diff_real + wi * diff_imag;
                    const FrameVec bin_real =
                        0.5 * (sum_real + correction_real);
                    const FrameVec bin_imag =
                        0.5 * (sum_imag + correction_imag);
                    power[k] = bin_real * bin_real + bin_imag * bin_imag;
                }
            } else {
                const FrameVec dc = real[0] + imag[0];
                const FrameVec nyquist = real[0] - imag[0];
                power[0] = dc * dc;
                power[half_fft] = nyquist * nyquist;

                for (std::size_t k = 1; k <= half_fft / 2; ++k) {
                    const std::size_t partner = half_fft - k;
                    const FrameVec left_real = real[k];
                    const FrameVec left_imag = imag[k];
                    const FrameVec right_real = real[partner];
                    const FrameVec right_imag = imag[partner];
                    const FrameVec sum_real = left_real + right_real;
                    const FrameVec sum_imag = left_imag - right_imag;
                    const FrameVec diff_real = left_real - right_real;
                    const FrameVec diff_imag = left_imag + right_imag;
                    const double wr = rfft_twiddle_real[k];
                    const double wi = rfft_twiddle_imag[k];
                    const FrameVec correction_real =
                        wr * diff_imag + wi * diff_real;
                    const FrameVec correction_imag =
                        -wr * diff_real + wi * diff_imag;
                    const FrameVec bin_real =
                        0.5 * (sum_real + correction_real);
                    const FrameVec bin_imag =
                        0.5 * (sum_imag + correction_imag);
                    power[k] = bin_real * bin_real + bin_imag * bin_imag;
                    if (partner != k) {
                        const FrameVec partner_real =
                            0.5 * (sum_real - correction_real);
                        const FrameVec partner_imag =
                            0.5 * (-sum_imag + correction_imag);
                        power[partner] = partner_real * partner_real +
                                         partner_imag * partner_imag;
                    }
                }
            }

            for (int m = 0; m < n_mels; ++m) {
                const MelBand& band = mel.bands[static_cast<std::size_t>(m)];
                FrameVec acc = zero_frame_vec();
                for (std::size_t k = band.first; k < band.end; ++k) {
                    acc += power[k] * mel.values[band.offset + k - band.first];
                }
                double* destination =
                    tile.data() + static_cast<std::size_t>(m) * kClaimFrames +
                    (frame - begin);
                for (std::size_t lane = 0; lane < count; ++lane) {
                    destination[lane] = frame_lane(acc, lane);
                }
            }
            frame += count;
        }
    };

    // Workers claim chunks of lane groups from a shared counter rather than
    // one contiguous range each: a static split left the slowest of 64
    // workers more than 50% behind the fastest (see report, round 4). As in
    // the MKL path, worker 0 sizes (zero-fills) the output while the others
    // already compute, keeping finished tiles in a private stash until the
    // output is published.
    const std::size_t claims = group_count / kGroupsPerClaim +
                               (group_count % kGroupsPerClaim != 0 ? 1 : 0);
    std::atomic<std::size_t> next_claim{0};
    enum : int { kPending, kReady, kFailed };
    std::atomic<int> output_state{kPending};
    double* output_data = nullptr;  // published by output_state
    run_workers(workers, [&](std::size_t worker) {
        if (worker == 0) {
            try {
                output.resize(output_count);
                output_data = output.data();
            } catch (...) {
                output_state.store(kFailed, std::memory_order_release);
                throw;
            }
            output_state.store(kReady, std::memory_order_release);
        }
        bool ready = worker == 0;
        std::vector<FrameVec> scratch(scratch_count);
        std::vector<double> tile(mel_count * kClaimFrames);
        std::vector<std::pair<std::size_t, std::vector<double>>> stash;
        const auto store = [&](std::size_t claim, const double* values) {
            const std::size_t first = claim * kClaimFrames;
            const std::size_t count = std::min(kClaimFrames, n_frames - first);
            for (std::size_t m = 0; m < mel_count; ++m)
                std::copy(values + m * kClaimFrames,
                          values + m * kClaimFrames + count,
                          output_data + m * n_frames + first);
        };
        // Returns false if the output could not be prepared.
        const auto become_ready = [&](bool wait) {
            int state = output_state.load(std::memory_order_acquire);
            while (wait && state == kPending) {
                std::this_thread::yield();
                state = output_state.load(std::memory_order_acquire);
            }
            if (state == kReady) {
                ready = true;
                for (const auto& entry : stash)
                    store(entry.first, entry.second.data());
                stash.clear();
            }
            return state != kFailed;
        };
        for (;;) {
            if (!ready && !become_ready(false)) return;
            const std::size_t claim =
                next_claim.fetch_add(1, std::memory_order_relaxed);
            if (claim >= claims) break;
            process_block(claim * kGroupsPerClaim,
                          std::min(group_count, (claim + 1) * kGroupsPerClaim),
                          scratch, tile);
            if (ready)
                store(claim, tile.data());
            else
                stash.emplace_back(claim, tile);
        }
        if (!ready) become_ready(true);
    });
}
