#include "melspec.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Parameters {
    int sr = 96000;
    int n_fft = 8192;
    int hop = 128;
    int n_mels = 256;
    double f_min = 0.0;
    double f_max = 22050.0;
};

double parse_double(const char* text, const char* name) {
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0')
        throw std::invalid_argument(std::string("invalid ") + name);
    return value;
}

Parameters parse_parameters(int argc, char** argv) {
    Parameters p;
    if (argc > 3) p.sr = std::stoi(argv[3]);
    if (argc > 4) p.n_fft = std::stoi(argv[4]);
    if (argc > 5) p.hop = std::stoi(argv[5]);
    if (argc > 6) p.n_mels = std::stoi(argv[6]);
    if (argc > 7) p.f_min = parse_double(argv[7], "f_min");
    if (argc > 8) p.f_max = parse_double(argv[8], "f_max");
    return p;
}

double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 9) {
        std::cerr << "usage: benchmark_compute PCM OUTPUT [SR N_FFT HOP N_MELS "
                     "F_MIN F_MAX]\n";
        return 2;
    }

    try {
        const Parameters p = parse_parameters(argc, argv);
        std::vector<double> input;
        const auto read_begin = std::chrono::steady_clock::now();
        read_pcm(argv[1], input);
        const auto read_end = std::chrono::steady_clock::now();

        std::vector<double> output;
        const auto compute_begin = std::chrono::steady_clock::now();
        compute_melspectrogram(input, p.sr, p.n_fft, p.hop, p.n_mels, p.f_min,
                               p.f_max, output);
        const auto compute_end = std::chrono::steady_clock::now();

        if (input.size() < static_cast<std::size_t>(p.n_fft))
            throw std::runtime_error("input is shorter than n_fft");
        const int frames =
            1 + static_cast<int>(
                    (input.size() - static_cast<std::size_t>(p.n_fft)) /
                    static_cast<std::size_t>(p.hop));
        const auto write_begin = std::chrono::steady_clock::now();
        write_output(argv[2], output, p.n_mels, frames);
        const auto write_end = std::chrono::steady_clock::now();

        std::cout << "{\"input_samples\":" << input.size()
                  << ",\"frames\":" << frames
                  << ",\"output_values\":" << output.size()
                  << ",\"read_normalize_seconds\":"
                  << seconds(read_end - read_begin) << ",\"compute_seconds\":"
                  << seconds(compute_end - compute_begin)
                  << ",\"floatcast_write_seconds\":"
                  << seconds(write_end - write_begin) << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "benchmark_compute: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
