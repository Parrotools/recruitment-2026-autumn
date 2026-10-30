#include "melspec.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

class ScopedEnvironment {
   public:
    ScopedEnvironment(const char* name, const char* value) : name_(name) {
        const char* current = std::getenv(name);
        if (current != nullptr) {
            had_previous_ = true;
            previous_ = current;
        }
        if (setenv(name, value, 1) != 0)
            throw std::runtime_error("setenv failed for " + name_);
    }

    ~ScopedEnvironment() {
        if (had_previous_) {
            (void)setenv(name_.c_str(), previous_.c_str(), 1);
        } else {
            (void)unsetenv(name_.c_str());
        }
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

   private:
    std::string name_;
    std::string previous_;
    bool had_previous_ = false;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const std::string& label) {
    const double tolerance = 2e-9 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(label + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

double reference_hz_to_mel(double frequency) {
    return 2595.0 * std::log10(1.0 + frequency / 700.0);
}

double reference_mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::vector<std::complex<double>> direct_dft(const std::vector<double>& frame) {
    const std::size_t n = frame.size();
    std::vector<std::complex<double>> result(n / 2 + 1);
    for (std::size_t k = 0; k <= n / 2; ++k) {
        double real = 0.0;
        double imag = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double angle = -2.0 * kPi * static_cast<double>(k * i) /
                                 static_cast<double>(n);
            real += frame[i] * std::cos(angle);
            imag += frame[i] * std::sin(angle);
        }
        result[k] = {real, imag};
    }
    return result;
}

std::vector<double> reference_filterbank(int sr, int n_fft, int n_mels,
                                         double f_min, double f_max) {
    const std::size_t n_bins = static_cast<std::size_t>(n_fft / 2) + 1;
    const double mel_min = reference_hz_to_mel(f_min);
    const double mel_max = reference_hz_to_mel(f_max);
    std::vector<double> points(static_cast<std::size_t>(n_mels) + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double mel = mel_min + static_cast<double>(i) *
                                         (mel_max - mel_min) /
                                         static_cast<double>(n_mels + 1);
        points[static_cast<std::size_t>(i)] = reference_mel_to_hz(mel);
    }

    std::vector<double> weights(static_cast<std::size_t>(n_mels) * n_bins, 0.0);
    for (int m = 0; m < n_mels; ++m) {
        const double left = points[static_cast<std::size_t>(m)];
        const double center = points[static_cast<std::size_t>(m) + 1];
        const double right = points[static_cast<std::size_t>(m) + 2];
        const double normalization = 2.0 / (right - left);
        for (std::size_t k = 0; k < n_bins; ++k) {
            const double frequency = static_cast<double>(k) *
                                     static_cast<double>(sr) /
                                     static_cast<double>(n_fft);
            const double rising = (frequency - left) / (center - left);
            const double falling = (right - frequency) / (right - center);
            weights[static_cast<std::size_t>(m) * n_bins + k] =
                std::max(0.0, std::min(rising, falling)) * normalization;
        }
    }
    return weights;
}

std::vector<double> reference_melspectrogram(const std::vector<double>& input,
                                             int sr, int n_fft, int hop,
                                             int n_mels, double f_min,
                                             double f_max,
                                             std::size_t n_frames) {
    const std::size_t n_bins = static_cast<std::size_t>(n_fft / 2) + 1;
    const std::vector<double> weights =
        reference_filterbank(sr, n_fft, n_mels, f_min, f_max);
    std::vector<double> output(static_cast<std::size_t>(n_mels) * n_frames,
                               0.0);
    std::vector<double> window(static_cast<std::size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        window[static_cast<std::size_t>(i)] =
            0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) /
                                  static_cast<double>(n_fft)));
    }

    for (std::size_t frame_index = 0; frame_index < n_frames; ++frame_index) {
        const std::size_t start = frame_index * static_cast<std::size_t>(hop);
        std::vector<double> frame(static_cast<std::size_t>(n_fft));
        for (int i = 0; i < n_fft; ++i) {
            frame[static_cast<std::size_t>(i)] =
                input[start + static_cast<std::size_t>(i)] *
                window[static_cast<std::size_t>(i)];
        }
        const auto spectrum = direct_dft(frame);
        for (int m = 0; m < n_mels; ++m) {
            double total = 0.0;
            for (std::size_t k = 0; k < n_bins; ++k) {
                const double real = spectrum[k].real();
                const double imag = spectrum[k].imag();
                total += weights[static_cast<std::size_t>(m) * n_bins + k] *
                         (real * real + imag * imag);
            }
            output[static_cast<std::size_t>(m) * n_frames + frame_index] =
                total;
        }
    }
    return output;
}

void check_rfft_sizes() {
    std::mt19937_64 random(1203);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (int n : {1, 2, 4, 8, 32}) {
        std::vector<double> input(static_cast<std::size_t>(n));
        for (double& value : input) value = distribution(random);
        std::vector<std::complex<double>> actual;
        compute_rfft(input, actual);
        const auto expected = direct_dft(input);
        require(actual.size() == expected.size(), "RFFT output size mismatch");
        for (std::size_t k = 0; k < actual.size(); ++k) {
            require_close(actual[k].real(), expected[k].real(), "RFFT real");
            require_close(actual[k].imag(), expected[k].imag(), "RFFT imag");
        }
    }
}

void check_mel_cases() {
    ScopedEnvironment threads("MELSPEC_THREADS", "1");
    std::mt19937_64 random(7823);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    const int sr = 32;
    for (int n_fft : {1, 2, 4, 8, 16, 32, 64, 128}) {
        for (std::size_t n_frames :
             {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 15u, 16u, 17u, 18u}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                const int hop = std::max(1, n_fft / 2);
                const int n_mels = pattern == 0 ? 3 : (n_fft < 16 ? 8 : 12);
                const double f_min = pattern == 1 ? 2.0 : 0.0;
                const double f_max = pattern == 2 ? 28.0 : 16.0;
                const std::size_t sample_count =
                    (n_frames - 1) * static_cast<std::size_t>(hop) +
                    static_cast<std::size_t>(n_fft);
                std::vector<double> input(sample_count, 0.0);
                if (pattern == 1 && !input.empty()) {
                    input[input.size() / 2] = 1.0;
                } else if (pattern == 2) {
                    for (double& value : input) value = distribution(random);
                }

                std::vector<double> actual;
                compute_melspectrogram(input, sr, n_fft, hop, n_mels, f_min,
                                       f_max, actual);
                const auto expected = reference_melspectrogram(
                    input, sr, n_fft, hop, n_mels, f_min, f_max, n_frames);
                require(actual.size() == expected.size(),
                        "Mel output size mismatch");
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    require_close(actual[i], expected[i], "Mel output");
                }

                const auto dense =
                    build_mel_filterbank(sr, n_fft, n_mels, f_min, f_max);
                const auto dense_reference =
                    reference_filterbank(sr, n_fft, n_mels, f_min, f_max);
                for (int m = 0; m < n_mels; ++m) {
                    for (std::size_t k = 0;
                         k < static_cast<std::size_t>(n_fft / 2 + 1); ++k) {
                        require_close(
                            dense[static_cast<std::size_t>(m)][k],
                            dense_reference
                                [static_cast<std::size_t>(m) *
                                     (static_cast<std::size_t>(n_fft / 2) + 1) +
                                 k],
                            "Mel filter weight");
                    }
                }
            }
        }
    }
}

void check_power_clipping_cases() {
    struct BandCase {
        double f_min;
        double f_max;
        int n_mels;
        const char* name;
    };
    const BandCase cases[] = {
        {0.0, 23.0, 5, "below sr/4"},
        {18.0, 30.0, 7, "cross sr/4"},
        {30.0, 45.0, 5, "high-frequency only"},
        {44.0, 52.0, 5, "Nyquist and above"},
        {50.0, 60.0, 4, "all bands empty"},
        {20.0, 22.0, 16, "partially empty bands"},
        {18.0, 27.0, 1, "half-spectrum center bin"},
    };
    std::mt19937_64 random(20261029);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    ScopedEnvironment threads("MELSPEC_THREADS", "3");

    for (const BandCase& test : cases) {
        const auto dense =
            build_mel_filterbank(96, 32, test.n_mels, test.f_min, test.f_max);
        if (std::string(test.name) == "all bands empty") {
            for (const auto& band : dense) {
                require(std::all_of(band.begin(), band.end(),
                                    [](double value) { return value == 0.0; }),
                        "expected a wholly empty mel range");
            }
        }
        if (std::string(test.name) == "partially empty bands") {
            bool has_empty = false;
            bool has_nonempty = false;
            for (const auto& band : dense) {
                const bool nonempty =
                    std::any_of(band.begin(), band.end(),
                                [](double value) { return value > 0.0; });
                has_nonempty |= nonempty;
                has_empty |= !nonempty;
            }
            require(has_empty && has_nonempty,
                    "expected both empty and nonempty mel filters");
        }
        if (std::string(test.name) == "half-spectrum center bin") {
            require(dense[0][8] > 0.0,
                    "test band must include the half-spectrum center bin");
            require(std::none_of(dense[0].begin() + 9, dense[0].end(),
                                 [](double value) { return value > 0.0; }),
                    "center-bin test must stay within lower-half cutoff");
        }

        for (std::size_t n_frames : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 17u}) {
            const std::size_t input_size =
                (n_frames - 1) * 16 + static_cast<std::size_t>(32);
            std::vector<double> input(input_size);
            for (double& value : input) value = distribution(random);
            std::vector<double> actual;
            compute_melspectrogram(input, 96, 32, 16, test.n_mels, test.f_min,
                                   test.f_max, actual);
            const auto expected =
                reference_melspectrogram(input, 96, 32, 16, test.n_mels,
                                         test.f_min, test.f_max, n_frames);
            require(actual.size() == expected.size(),
                    std::string(test.name) + ": output size mismatch");
            for (std::size_t i = 0; i < expected.size(); ++i) {
                require_close(actual[i], expected[i], test.name);
            }
        }
    }
}

void check_parallel_repeatability() {
    std::vector<double> input(1200);
    std::mt19937_64 random(44);
    std::uniform_real_distribution<double> distribution(-0.9, 0.9);
    for (double& value : input) value = distribution(random);

    std::vector<double> single;
    {
        ScopedEnvironment threads("MELSPEC_THREADS", "1");
        compute_melspectrogram(input, 16000, 32, 13, 17, 0.0, 9000.0, single);
    }
    std::vector<double> parallel;
    {
        ScopedEnvironment threads("MELSPEC_THREADS", "4");
        compute_melspectrogram(input, 16000, 32, 13, 17, 0.0, 9000.0, parallel);
    }
    require(single.size() == parallel.size(), "Parallel output size mismatch");
    for (std::size_t i = 0; i < single.size(); ++i) {
        require_close(parallel[i], single[i], "Parallel output");
    }
    {
        ScopedEnvironment threads("MELSPEC_THREADS", "4");
        std::vector<double> repeated;
        compute_melspectrogram(input, 16000, 32, 13, 17, 0.0, 9000.0, repeated);
        require(repeated == parallel, "Repeated parallel call changed output");
    }
}

void check_in_place_aliasing() {
    std::mt19937_64 random(9127);
    std::uniform_real_distribution<double> distribution(-0.8, 0.8);

    const auto check_case = [&](int n_fft, int hop, std::size_t n_frames,
                                int n_mels, double f_max,
                                const char* thread_count,
                                bool expect_output_smaller) {
        const std::size_t input_size =
            (n_frames - 1) * static_cast<std::size_t>(hop) +
            static_cast<std::size_t>(n_fft);
        std::vector<double> input(input_size);
        for (double& value : input) value = distribution(random);

        const auto expected = reference_melspectrogram(
            input, 32, n_fft, hop, n_mels, 0.0, f_max, n_frames);
        const std::size_t output_size =
            static_cast<std::size_t>(n_mels) * n_frames;
        require((output_size < input_size) == expect_output_smaller,
                "Alias test did not exercise the requested resize direction");

        std::vector<double> aliased = input;
        {
            ScopedEnvironment threads("MELSPEC_THREADS", thread_count);
            compute_melspectrogram(aliased, 32, n_fft, hop, n_mels, 0.0, f_max,
                                   aliased);
        }
        require(aliased.size() == expected.size(),
                "Aliased output size mismatch");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            require_close(aliased[i], expected[i], "Aliased Mel output");
        }
    };

    // Shrinking input: 9 frames use a 4-lane tail batch and 3 frame workers.
    check_case(32, 16, 9, 2, 28.0, "3", true);
    // Growing input: exercises a partial final SIMD batch and multiple workers.
    check_case(4, 2, 5, 12, 28.0, "3", false);
    // N=1 has a dedicated zero-window path and still must survive alias resize.
    check_case(1, 1, 7, 2, 28.0, "3", false);
}

void check_invalid_inputs() {
    const std::vector<double> audio(16, 0.25);
    const auto expect_throw = [](const std::function<void()>& action,
                                 const std::string& label) {
        bool threw = false;
        try {
            action();
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, "Expected exception for " + label);
    };

    std::vector<double> output;
    expect_throw(
        [&] { compute_melspectrogram(audio, 0, 8, 2, 4, 0, 4, output); },
        "zero sample rate");
    expect_throw(
        [&] { compute_melspectrogram(audio, 8, 0, 2, 4, 0, 4, output); },
        "zero FFT size");
    expect_throw(
        [&] { compute_melspectrogram(audio, 8, 6, 2, 4, 0, 4, output); },
        "non-power-of-two FFT size");
    expect_throw(
        [&] { compute_melspectrogram(audio, 8, 8, 0, 4, 0, 4, output); },
        "zero hop");
    expect_throw(
        [&] { compute_melspectrogram(audio, 8, 8, 2, 0, 0, 4, output); },
        "zero mel count");
    expect_throw(
        [&] { compute_melspectrogram(audio, 8, 8, 2, 4, -1, 4, output); },
        "negative minimum frequency");
    expect_throw(
        [&] { compute_melspectrogram(audio, 8, 8, 2, 4, 4, 4, output); },
        "empty frequency interval");
    expect_throw(
        [&] {
            compute_melspectrogram(audio, 8, 8, 2, 4, 0,
                                   std::numeric_limits<double>::infinity(),
                                   output);
        },
        "infinite maximum frequency");
    expect_throw(
        [&] {
            compute_melspectrogram(std::vector<double>(7), 8, 8, 2, 4, 0, 4,
                                   output);
        },
        "short input");
    expect_throw(
        [&] {
            std::vector<std::complex<double>> rfft;
            compute_rfft({}, rfft);
        },
        "empty RFFT");
    expect_throw(
        [&] {
            std::vector<std::complex<double>> rfft;
            compute_rfft(std::vector<double>(3), rfft);
        },
        "non-power-of-two RFFT");
    {
        ScopedEnvironment threads("MELSPEC_THREADS", "0");
        expect_throw(
            [&] { compute_melspectrogram(audio, 8, 8, 2, 4, 0, 4, output); },
            "invalid thread count");
    }
}

}  // namespace

int main() {
    try {
        check_rfft_sizes();
        check_mel_cases();
        check_power_clipping_cases();
        check_parallel_repeatability();
        check_in_place_aliasing();
        check_invalid_inputs();
    } catch (const std::exception& error) {
        std::cerr << "test_melspec: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test_melspec: PASS\n";
    return 0;
}
