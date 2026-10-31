#pragma once
// Optional CUDA path (built when MELSPEC_CUDA=1). Internal to melspec.cpp.
#include <cstddef>
#include <functional>

namespace melspec_cuda {

// Waits for the start-up thread that creates the CUDA context (it starts
// before main, so this overlaps main.cpp's input read) and reports whether a
// usable device was found. Never throws.
bool available() noexcept;

// Pinned staging buffers that compute() cycles through for its results.
constexpr std::size_t kSlots = 4;

struct Job {
    const float* samples;
    std::size_t n_samples;
    std::size_t n_frames;
    std::size_t fft_size;
    std::size_t hop;
    std::size_t n_mels;
    // Compact Mel weights: band m covers bins [first[m], end[m]) and its
    // weights start at weights[offset[m]].
    const std::size_t* first;
    const std::size_t* end;
    const std::size_t* offset;
    const double* weights;
    std::size_t n_weights;
};

// Computes the Mel spectrogram on the GPU in chunks of frames, calling
// `consume(frame0, count, rows)` once per chunk, in order, with rows laid out
// [mel][count] in float. `rows` stays valid only until consume returns.
// Throws std::runtime_error on CUDA failures.
void compute(const Job& job,
             const std::function<void(std::size_t, std::size_t,
                                      const float*)>& consume,
             bool trace);

}  // namespace melspec_cuda
