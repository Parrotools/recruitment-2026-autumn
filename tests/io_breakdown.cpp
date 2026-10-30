// Times the individual steps of main.cpp's read_pcm/write_output (copied
// verbatim, with timers between statements) so allocator changes can be
// judged on the part of the run that compute_melspectrogram does not own.
//
// Usage: io_breakdown INPUT_PCM OUTPUT_BIN N_VALUES
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point& t) {
    const Clock::time_point now = Clock::now();
    const double s = std::chrono::duration<double>(now - t).count();
    t = now;
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    Clock::time_point t = Clock::now();
    std::ifstream f(argv[1], std::ios::binary);
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    int n_samples = static_cast<int>(size / sizeof(int16_t));
    std::vector<int16_t> buf(n_samples);
    const double buf_alloc = since(t);
    f.read(reinterpret_cast<char*>(buf.data()), size);
    const double file_read = since(t);
    std::vector<double> y;
    y.resize(n_samples);
    const double y_alloc = since(t);
    for (int i = 0; i < n_samples; ++i)
        y[i] = static_cast<double>(buf[i]) / 32768.0;
    const double convert_in = since(t);

    const std::size_t n = std::strtoull(argv[3], nullptr, 10);
    std::vector<double> data(n, 1.0);
    since(t);

    std::ofstream o(argv[2], std::ios::binary);
    const double open_out = since(t);
    std::vector<float> out_f32(data.size());
    const double out_alloc = since(t);
    for (size_t i = 0; i < data.size(); ++i)
        out_f32[i] = static_cast<float>(data[i]);
    const double convert_out = since(t);
    o.write(reinterpret_cast<const char*>(out_f32.data()),
            static_cast<std::streamsize>(out_f32.size() * sizeof(float)));
    o.flush();
    const double file_write = since(t);
    std::printf(
        "buf_alloc %.4f file_read %.4f y_alloc %.4f convert_in %.4f | "
        "open_out %.4f out_alloc %.4f convert_out %.4f file_write %.4f\n",
        buf_alloc, file_read, y_alloc, convert_in, open_out, out_alloc,
        convert_out, file_write);
    return y[0] > 2.0 ? 1 : 0;
}
