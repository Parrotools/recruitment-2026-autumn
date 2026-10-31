// CUDA path: cuFFT real transforms straight from the samples, then the
// frequency-domain Hann window, power and Mel bands in one kernel. Float
// throughout: consumer and workstation GPUs run double at 1/64 rate.
#include "melspec_cuda.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace melspec_cuda {
namespace {

// Size of each of the kSlots pinned staging buffers for the device-to-host
// copies of Mel rows.
constexpr std::size_t kSlotBytes = std::size_t(8) << 20;
// Frames per chunk at most, and the device memory for one chunk's spectrum.
constexpr std::size_t kMaxChunkFrames = 16384;
constexpr std::size_t kSpectrumBudget = std::size_t(1) << 30;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA ") + what + ": " +
                                 cudaGetErrorString(status));
}

void check(cufftResult status, const char* what) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("cuFFT ") + what + " failed (" +
                                 std::to_string(static_cast<int>(status)) +
                                 ")");
}

// Creating a CUDA context takes about 0.1 s, as long as main.cpp's input
// read; a thread started during static initialization does it in parallel.
class Startup {
   public:
    // MELSPEC_NO_CUDA (any value) skips the GPU entirely, for comparisons.
    Startup() {
        if (std::getenv("MELSPEC_NO_CUDA") == nullptr)
            thread_ = std::thread([this] { run(); });
    }
    ~Startup() {
        if (thread_.joinable()) thread_.join();
    }
    bool wait() noexcept {
        if (thread_.joinable()) thread_.join();
        return ok_;
    }
    float* slot(std::size_t index) const { return slots_[index]; }

   private:
    void run() noexcept {
        const auto start = std::chrono::steady_clock::now();
        const auto since = [&] {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                .count();
        };
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return;
        const double counted = since();
        if (cudaFree(nullptr) != cudaSuccess) return;
        const double context = since();
        for (float*& slot : slots_)
            if (cudaMallocHost(&slot, kSlotBytes) != cudaSuccess) return;
        ok_ = true;
        if (std::getenv("MELSPEC_TRACE") != nullptr)
            std::fprintf(stderr,
                         "trace cuda startup: devices %.4f context %.4f "
                         "pinned %.4f s\n",
                         counted, context, since());
    }

    float* slots_[kSlots] = {};
    bool ok_ = false;
    std::thread thread_;
};

Startup g_startup;

template <typename T>
struct DeviceArray {
    T* data = nullptr;
    explicit DeviceArray(std::size_t count) {
        check(cudaMalloc(&data, std::max<std::size_t>(count, 1) * sizeof(T)),
              "cudaMalloc");
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

struct Resources {
    cudaStream_t upload = nullptr;
    cudaStream_t compute = nullptr;
    cudaStream_t copy = nullptr;
    cufftHandle plan = 0;
    bool has_plan = false;
    std::vector<cudaEvent_t> events;
    ~Resources() {
        for (cudaStream_t stream : {upload, compute, copy})
            if (stream != nullptr) cudaStreamSynchronize(stream);
        if (has_plan) cufftDestroy(plan);
        for (cudaEvent_t event : events) cudaEventDestroy(event);
        for (cudaStream_t stream : {upload, compute, copy})
            if (stream != nullptr) cudaStreamDestroy(stream);
    }
    cudaEvent_t event() {
        cudaEvent_t event;
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
              "cudaEventCreate");
        events.push_back(event);
        return event;
    }
};

// The periodic Hann window is w[n] = 1/2 - (e^{2 pi i n/N} + e^{-2 pi i
// n/N})/4, so the windowed frame's DFT is Y[k] = X[k]/2 - (X[k-1]+X[k+1])/4,
// with X[-1] = conj(X[1]) and X[N/2+1] = conj(X[N/2-1]) for real input.
// spectrum is cuFFT's [frame][bin] output (n_bins per frame); power is
// written transposed, [bin - active_first][frame], through a shared tile so
// that both the loads here and mel_bands' loads are contiguous in a warp.
constexpr int kTile = 32;
__global__ void window_power(const float2* spectrum, int n_bins,
                             int active_first, int active_bins, int count,
                             int stride, float* power) {
    __shared__ float tile[kTile][kTile + 1];
    const int b = blockIdx.x * kTile + threadIdx.x;  // active bin index
    const int k = active_first + b;
    const int half = n_bins - 1;
    for (int y = threadIdx.y; y < kTile; y += blockDim.y) {
        const int f = blockIdx.y * kTile + y;
        float value = 0.0f;
        if (b < active_bins && f < count) {
            const float2* x = spectrum + std::size_t(f) * n_bins;
            const float2 c = x[k];
            float2 p = k > 0 ? x[k - 1] : make_float2(x[1].x, -x[1].y);
            float2 n = k < half ? x[k + 1]
                                : make_float2(x[half - 1].x, -x[half - 1].y);
            const float re = 0.5f * c.x - 0.25f * (p.x + n.x);
            const float im = 0.5f * c.y - 0.25f * (p.y + n.y);
            value = re * re + im * im;
        }
        tile[y][threadIdx.x] = value;
    }
    __syncthreads();
    for (int y = threadIdx.y; y < kTile; y += blockDim.y) {
        const int bin = blockIdx.x * kTile + y;
        const int f = blockIdx.y * kTile + threadIdx.x;
        if (bin < active_bins && f < count)
            power[std::size_t(bin) * stride + f] = tile[threadIdx.x][y];
    }
}

// One thread per frame; out is this chunk's [mel][count] block.
__global__ void mel_bands(const float* power, int stride, int active_first,
                          const int* first, const int* end, const int* offset,
                          const float* weights, int n_mels, float* out,
                          int count) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= count) return;
    const float* p = power + f - std::size_t(active_first) * stride;
    for (int m = 0; m < n_mels; ++m) {
        const int k0 = first[m];
        const int k1 = end[m];
        const float* w = weights + offset[m] - k0;
        float sum = 0.0f;
        for (int k = k0; k < k1; ++k)
            sum = fmaf(p[std::size_t(k) * stride], w[k], sum);
        out[std::size_t(m) * count + f] = sum;
    }
}

}  // namespace

bool available() noexcept { return g_startup.wait(); }

void compute(const Job& job,
             const std::function<void(std::size_t, std::size_t,
                                      const float*)>& consume,
             bool trace) {
    using Clock = std::chrono::steady_clock;
    const Clock::time_point start = Clock::now();
    const auto seconds = [&] {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    if (!available()) throw std::runtime_error("no usable CUDA device");

    const std::size_t n_bins = job.fft_size / 2 + 1;
    std::size_t chunk = std::min(kMaxChunkFrames, job.n_frames);
    chunk = std::min(chunk, kSlotBytes / (job.n_mels * sizeof(float)));
    chunk = std::min(chunk, kSpectrumBudget / (n_bins * sizeof(float2)));
    if (chunk == 0 || job.fft_size > INT_MAX || job.hop > INT_MAX ||
        job.n_frames > static_cast<std::size_t>(INT_MAX))
        throw std::runtime_error("problem size unsupported by the CUDA path");
    const std::size_t chunks = (job.n_frames + chunk - 1) / chunk;
    // The last chunk transforms whole `chunk` frames; pad the samples with
    // zeros so its unused frames stay in bounds.
    const std::size_t padded =
        std::max(job.n_samples, (chunks * chunk - 1) * job.hop + job.fft_size);

    std::size_t active_first = n_bins, active_end = 0;
    for (std::size_t m = 0; m < job.n_mels; ++m) {
        if (job.first[m] == job.end[m]) continue;
        active_first = std::min(active_first, job.first[m]);
        active_end = std::max(active_end, job.end[m]);
    }
    if (active_end <= active_first) active_first = active_end = 0;
    const std::size_t active_bins = active_end - active_first;

    Resources res;
    for (cudaStream_t* stream : {&res.upload, &res.compute, &res.copy})
        check(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking),
              "cudaStreamCreate");

    DeviceArray<float> input(padded);
    DeviceArray<float2> spectrum(n_bins * chunk);
    DeviceArray<float> power(active_bins * chunk);
    DeviceArray<float> out(job.n_mels * job.n_frames);
    DeviceArray<int> bands(3 * job.n_mels);
    DeviceArray<float> weights(job.n_weights);

    std::vector<int> band_host(3 * job.n_mels);
    for (std::size_t m = 0; m < job.n_mels; ++m) {
        band_host[m] = static_cast<int>(job.first[m]);
        band_host[job.n_mels + m] = static_cast<int>(job.end[m]);
        band_host[2 * job.n_mels + m] = static_cast<int>(job.offset[m]);
    }
    std::vector<float> weight_host(job.weights, job.weights + job.n_weights);
    check(cudaMemcpyAsync(bands.data, band_host.data(),
                          band_host.size() * sizeof(int),
                          cudaMemcpyHostToDevice, res.compute),
          "band upload");
    check(cudaMemcpyAsync(weights.data, weight_host.data(),
                          weight_host.size() * sizeof(float),
                          cudaMemcpyHostToDevice, res.compute),
          "weight upload");
    if (padded > job.n_samples)
        check(cudaMemsetAsync(input.data + job.n_samples, 0,
                              (padded - job.n_samples) * sizeof(float),
                              res.compute),
              "pad");


    int n = static_cast<int>(job.fft_size);
    int in_embed = n;
    int out_embed = static_cast<int>(n_bins);
    check(cufftPlanMany(&res.plan, 1, &n, &in_embed, 1,
                        static_cast<int>(job.hop), &out_embed, 1,
                        static_cast<int>(n_bins), CUFFT_R2C,
                        static_cast<int>(chunk)),
          "plan");
    res.has_plan = true;
    check(cufftSetStream(res.plan, res.compute), "set stream");
    const double planned = seconds();

    // Samples are uploaded just ahead of the first chunk that reads them, on
    // their own stream, so the copies overlap the transforms of earlier
    // chunks. (Pageable copies: pinning the input would cost more than it
    // saves.)
    std::size_t uploaded = 0;
    const auto upload_until = [&](std::size_t needed) {
        constexpr std::size_t kPiece = std::size_t(1) << 21;
        needed = std::min(needed, job.n_samples);
        while (uploaded < needed) {
            const std::size_t count =
                std::min(job.n_samples - uploaded, std::max(kPiece, needed - uploaded));
            check(cudaMemcpyAsync(input.data + uploaded,
                                  job.samples + uploaded,
                                  count * sizeof(float),
                                  cudaMemcpyHostToDevice, res.upload),
                  "sample upload");
            uploaded += count;
        }
        cudaEvent_t event = res.event();
        check(cudaEventRecord(event, res.upload), "event record");
        check(cudaStreamWaitEvent(res.compute, event, 0), "wait event");
    };

    std::vector<cudaEvent_t> computed(chunks);
    for (std::size_t c = 0; c < chunks; ++c) {
        const std::size_t frame0 = c * chunk;
        const int count = static_cast<int>(std::min(chunk, job.n_frames - frame0));
        float* chunk_out = out.data + frame0 * job.n_mels;
        upload_until((frame0 + count - 1) * job.hop + job.fft_size);
        check(cufftExecR2C(res.plan, input.data + frame0 * job.hop,
                           reinterpret_cast<cufftComplex*>(spectrum.data)),
              "exec");
        if (active_bins > 0) {
            const dim3 tiles(static_cast<unsigned>((active_bins + kTile - 1) / kTile),
                             static_cast<unsigned>((count + kTile - 1) / kTile));
            window_power<<<tiles, dim3(kTile, 8), 0, res.compute>>>(
                spectrum.data, static_cast<int>(n_bins),
                static_cast<int>(active_first), static_cast<int>(active_bins),
                count, static_cast<int>(chunk), power.data);
            check(cudaGetLastError(), "power launch");
        }
        mel_bands<<<(count + 255) / 256, 256, 0, res.compute>>>(
            power.data, static_cast<int>(chunk),
            static_cast<int>(active_first), bands.data,
            bands.data + job.n_mels, bands.data + 2 * job.n_mels,
            weights.data, static_cast<int>(job.n_mels), chunk_out, count);
        check(cudaGetLastError(), "mel launch");
        computed[c] = res.event();
        check(cudaEventRecord(computed[c], res.compute), "event record");
    }
    const double submitted = seconds();

    cudaEvent_t copied[kSlots];
    for (cudaEvent_t& event : copied) event = res.event();
    const auto copy_chunk = [&](std::size_t c) {
        const std::size_t frame0 = c * chunk;
        const std::size_t count = std::min(chunk, job.n_frames - frame0);
        check(cudaStreamWaitEvent(res.copy, computed[c], 0), "wait event");
        check(cudaMemcpyAsync(g_startup.slot(c % kSlots),
                              out.data + frame0 * job.n_mels,
                              count * job.n_mels * sizeof(float),
                              cudaMemcpyDeviceToHost, res.copy),
              "result copy");
        check(cudaEventRecord(copied[c % kSlots], res.copy), "event record");
    };
    for (std::size_t c = 0; c < std::min(chunks, kSlots); ++c)
        copy_chunk(c);
    double first_ready = 0;
    for (std::size_t c = 0; c < chunks; ++c) {
        check(cudaEventSynchronize(copied[c % kSlots]), "result wait");
        if (c == 0) first_ready = seconds();
        const std::size_t frame0 = c * chunk;
        consume(frame0, std::min(chunk, job.n_frames - frame0),
                g_startup.slot(c % kSlots));
        if (c + kSlots < chunks) copy_chunk(c + kSlots);
    }
    if (trace)
        std::fprintf(stderr,
                     "trace cuda chunks %zu x %zu frames: plan %.4f "
                     "submitted %.4f first chunk %.4f done %.4f s\n",
                     chunks, chunk, planned, submitted, first_ready,
                     seconds());
}

}  // namespace melspec_cuda
