# Mel 频谱实现优化记录

## 当前状态与结论

| 轮次 | 当前代码与证据 | 状态 |
|---|---|---|
| R1 基础实现 | 沿用 heptagon 已有的半长实数 FFT、twiddle/bit-reversal 预计算、稀疏 Mel 区间、融合加窗装载、跨帧 SIMD 和线程帧块；补充参数边界、异常 join、别名输入输出和 allocator 生命周期处理。 | 已进入当前实现；R1/局部修复证据详见本报告前半部分。 |
| R2 候选 B | 按实际 Mel 频带范围裁剪无需读取的高频对称 power 重构；其余情况走完整重构。源文件 SHA256 `0734a650a461aa271a2c4555d3b93e16b399ca78b68e03f2887d882077db7b29`。 | 已通过集群 job 48 的四组正式 expected 校验及 job 51 的七轮端到端测量。8 核 hepnode3 上，AVX2 组 narrow 的 round1/B 配对比为 `1.03391`，HUGE_ALLOC 组为 `1.04917`；这是配对耗时比，1 表示相同，值大于 1 表示 B 耗时较低。 |
| R2 allocator | `HUGE_ALLOC=1` 的 Linux 全局大块分配器。 | 保持默认关闭。已测 binary `round2/cluster/build/validation-48/B_huge/melspec` SHA256 `ed9dcd655f82d14fffe291db48edc97bec67cefaaecda02fd0a807c00cdaccd5`；其 correctness 与性能证据适用于当时 hepnode3/GCC 配置，不代表其他机器。 |
| R3 候选 F/V | F 为相邻两级 radix-2 FFT 融合；V 为显式 AVX-512F、8 帧 lane；FV 合用。当前源码 SHA256 `712867d887e9af3b214806079ef72aa2220adddafeec08017359772d1271cce5`，实验开关均默认关闭。 | 五种候选配置只完成构建和定向 unit-test；尚无候选 NumPy 全集、正式 expected、四 case 筛选或确认轮性能结论。R3 tester 尚未提交正式计算作业。 |
| R4 | 用满 64 核；MKL DFTI 单帧 r2c（检测到 MKL 时默认启用，静态顺序链接）；原子计数器动态 tile 调度；行布局 power + `[mel][16]` tile 输出；输出清零与计算重叠；`write_output` 的 float 副本在 compute 中预触页。`HUGE_ALLOC`、`-march=native` 在 Linux 默认开启。 | 集群 hepnode3 独占 64 核（job 207），`validate.py --runs 5` 两轮四组全部 PASS，误差与 R2 逐位相同。default/small_mels/large_fft/narrow_band 为 **0.307/0.188/0.403/0.384 s**（R2 B_huge 同条件 0.370/0.226/0.450/0.442 s）。详见文末“第四轮”。 |
| **R5 当前默认** | 在 R4 基础上：周期 Hann 窗移到频域，DFTI out-of-place 直接读输入样本，`Y[k] = X[k]/2 - (X[k-1]+X[k+1])/4` 只在活跃 bin 上计算（开关 `MELSPEC_FREQ_WINDOW`，默认 1）；Mel 点积 4 帧一组累加（结果逐位不变）；`output` 的 resize 改由 worker 1 执行，与主线程预留 float 副本并行。绑核、NUMA 本地输出、分配器偏好本地节点、PERM 打包格式、两帧合并复数 FFT 等尝试均未采用（见第五轮）。 | job 235 独占 64 核，`validate --runs 5` 三轮四组全部 PASS，误差与 R4 逐位相同：default **0.284–0.287 s**、small_mels **0.179–0.183 s**、large_fft **0.359–0.368 s**、narrow_band **0.347–0.355 s**（R4 最终验收 job 207：0.307 / 0.188 / 0.403 / 0.384 s）。 |
| R6 / **R7** | R6：全 worker 分阶段插桩（`MELSPEC_TRACE=1`）；8 帧 Mel、DFTI 不用工作区、两级创建线程、共享描述符、tile 形状、`main.cpp` 编译选项都在满载下测过，均未采用。R7：没有 MKL 时的回退路径改为动态领取 + `[mel][32]` tile 输出 + resize 与计算重叠。 | 默认（MKL）构建的输出与 R5 逐字节相同、性能不变。回退路径（`MKL=0`，job 255）validate default/small_mels/large_fft/narrow_band 从 0.35/0.23/0.45/0.46 s 降到 **0.32–0.34/0.215–0.219/0.425–0.439/0.412–0.426 s**，输出逐字节不变。 |

第五轮之后，默认构建（`make`）就是上表 R5 行的配置：在集群上 Makefile 自动找到 spack 中的 oneMKL 并启用 MKL 路径，同时默认开启 `HUGE_ALLOC=1`（Linux）和 `-march=native`；没有 MKL 的机器自动退回 R1–R3 的手写 FFT 路径。R3 的 `FUSED_FFT`/`AVX512` 仍是默认关闭的实验开关，只作用于手写路径。float32 FFT（`CPPFLAGS=-DMELSPEC_FFT_FLOAT=1`）能再快 10–25%，但不是默认，原因见第四轮结论。下文第一到第三轮按当时的状态记录，其中“默认关闭 HUGE_ALLOC”等表述指的是当时的构建默认值。

## 范围和实现摘要

核心实现修改 src/melspec.cpp 与构建配置，另新增独立测试和可选 Linux 大块分配器；没有修改 src/main.cpp、validate.py 或 data。实现仍遵循现有接口，内部计算保持 double，最终输出仍由原有主程序写成 little-endian float32。

主要改动：

- 将每帧的完整复数 FFT 改为长度 n_fft/2 的实数打包 FFT：偶数样本放实部，奇数样本放虚部，再重构非负频率功率谱。
- 预计算 FFT twiddle、bit-reversal 次序、实数 FFT 重构 twiddle 和周期 Hann 窗。
- Mel filterbank 只保存每个滤波器覆盖的连续 bin 区间，并将权重平铺；按频率升序累计，空区间直接写零。
- 融合 Hann 乘法和 bit-reversal 装载；每小批帧完成 FFT、功率谱和 Mel 累计，避免保存全部帧的功率谱。
- SIMD 跨相邻帧运行，单帧内部的 bin 累加顺序不变。编译目标启用 x86 AVX2 时使用 4 个 double lane，在 ARM64 使用 2 个 lane，其他平台使用标量；SCALAR=1 可强制标量。
- 将连续的帧组分给 std::thread。MELSPEC_THREADS 必须是正整数，实际线程数受可用 CPU 和帧组数限制；Linux 优先按 sched_getaffinity 统计进程可用 CPU。
- 全局大块分配器设为 Linux 可选项，默认关闭。它只替换普通对齐的 new/new[]/delete/delete[]；SIMD 暂存使用 C++17 对齐分配。
- 跟进正确性审查：malloc 与 mmap 返回的存储均使用 placement new 显式开始 AllocationHeader 生命周期；输入与输出引用同一个 vector 时，在 resize 前复制输入，普通非别名路径不复制。

## 数学等价性

对长度为 N 的实数帧 x，先应用周期 Hann 窗 w[n] = 0.5(1-cos(2πn/N))。令 M=N/2，构造 z[m]=x[2m]w[2m] + i x[2m+1]w[2m+1]，计算 M 点复数 FFT Z，再用 Z[k] 和 conj(Z[M-k]) 重构原始 N 点实数 FFT 的非负频率。这样与直接对窗后实数帧计算 N 点 DFT 等价；N=1 的周期 Hann 为零，N=2 由单个打包元素直接生成 DC 和 Nyquist。

Mel 权重仍是 HTK mel 等间距点构造的三角滤波器，并使用原有 Slaney 面积归一化 2/(f_r-f_l)。紧凑存储只略过三角形支撑区间之外的零权重 bin，保留升序 bin 累加，因此不改变每个 Mel 输出的数学求和。输出仍为 [n_mels, n_frames] 行主序。

## 与 heptagon/melspec 参考实现的差别

参考目录：/Users/zigeqiao/workspace/heptagon/melspec/，参考提交 03d9542。参考 src/melspec.cpp SHA256 为 752e1be9d9dbc7521d667cb507172ba1558f07765460fbb5503023cbd3112d1c。 本仓库优化前基线来自提交 be4db00a4241b9e0cd401cd1328c060b792c0828，原始 src/melspec.cpp SHA256 为 52720604c94ac37a0a7ba25d90f376c4297f8eeaef50ca75e2d906c296621f48。

| 技术 | heptagon 参考 | 本次实现 |
|---|---|---|
| FFT twiddle 与 bit-reversal | 已预计算 | 预计算并在每个帧块复用 |
| 实数 FFT | n_fft/2 偶奇打包与重构 | 保留同一算法，补齐 N=1/2 和通用中心 bin 处理 |
| power 重构 | 完整生成后续 Mel 所需的 one-sided bins | R2 B 依据所有 Mel band 的有效 bin 包围区间裁剪高频对称部分；安全条件不满足时完整回退 |
| Mel 权重 | 连续稀疏区间及扁平权重 | 同样紧凑存储；校验 mel 点严格递增，空带安全写零 |
| 加窗与装载 | 融合 Hann 和 bit-reversal 装载 | 保留融合，末批复制最后有效帧，仅写真实帧 |
| 内存 | 逐块产生功率谱和 Mel 输出 | 保留分块处理，避免全量功率谱；size_t 索引并检查输出元素数溢出 |
| SIMD | 无条件 GCC AVX2/FMA target pragma，固定 4 lane | 根据编译目标选择 AVX2 4 lane、ARM64 2 lane 或标量；提供 SCALAR=1 |
| FFT stage 与宽度扩展 | 无相邻 radix-2 stage 融合或 8 lane AVX-512 变体 | R3 增加默认关闭的 FUSED_FFT 与 AVX512 实验开关；不是参考已有优化，也未验证性能收益 |
| 线程 | 按 CPU affinity 分连续帧组 | 保留 affinity 和连续分块；支持 MELSPEC_THREADS，捕获 worker 异常并确保 join 后再抛出 |
| 全局分配器 | 参考中默认启用 Linux mmap/THP 和触页 | HUGE_ALLOC=1 显式启用；默认关闭，非 Linux 构建时明确拒绝 |

参考 allocator 的映射长度、对齐余量和 header 加法没有完整溢出保护；小块分配没有显式处理零长度；mmap 失败直接抛错而没有 malloc 回退；自定义 new 没有保留 new_handler 重试语义。参考分配器还按所有可用 CPU 切分触页区间，却只启动受页数限制的部分线程，页数少于 CPU 数时可能留下未触碰区间；线程启动异常也可能在仍有 joinable 线程时退出。当前可选实现对尺寸加法与 2 MiB round-up 做边界检查，普通 new 保持默认对齐和 new_handler 语义，mmap 不可用时回退 malloc，按实际 worker 数切分页区，并在启动失败后 join 已启动线程、串行补触全部有效映射。THP 建议失败不影响正确性。

## 构建与测试入口

切换 SCALAR、ARCH_FLAGS 或 HUGE_ALLOC 等编译选项后先运行 make clean。

当前 Makefile 提供 `SCALAR`、`HUGE_ALLOC`、`FUSED_FFT`、`AVX512` 四个 0/1 选项，默认均为 0；改变任一选项后先 `make clean`。`HUGE_ALLOC=1` 仅支持 Linux。`FUSED_FFT=1` 选择 R3 F。`AVX512=1` 选择 R3 V，要求 `SCALAR=0`、编译器支持且 flags 包含 `-mavx512f`，运行主机也必须支持 AVX-512F；当前没有运行时 dispatch，默认不自动启用 AVX-512。`SCALAR=1` 优先选择标量实现。比如在 Slurm compute node 上构建 F 或 V：

    cd /home/qzg/workspace/melspec-optimization-20261029/round3/candidate-source
    make clean && make CXX=g++ ARCH_FLAGS="-mavx2 -mfma" FUSED_FFT=1
    make clean && make CXX=g++ ARCH_FLAGS="-mavx2 -mfma -mavx512f" AVX512=1

上述命令须在 Slurm compute allocation 中运行，目录使用 R3 `candidate-source` 或等价的当前 R3 checkout。不要在 login 节点编译，也不要在 `round2/cluster/repo` 使用 R3 选项；该目录的 Makefile 是旧快照，不含这些开关。这两条命令只产生候选构建；候选是否应被保留或启用，须看 R3 独立正确性与性能验收结果。

Prometheus 上确认的 x86 AVX2 构建命令为 make clean && make CXX=g++ ARCH_FLAGS="-mavx2 -mfma"。Linux 的可选分配器用 make clean && make CXX=g++ HUGE_ALLOC=1 ARCH_FLAGS="-mavx2 -mfma" 启用；HUGE_ALLOC 默认仍为 0。R2 job 48 已用正式输入验证 HUGE_ALLOC 正确性；job 51 的七轮比较是同为 HUGE_ALLOC 的 R1/B 版本比较，不是 allocator 开关 on/off 对照。R3 job 121 对 B、B+HUGE_ALLOC 和 reference 做了正式四 case 同组三轮初测，详见 R3 表；三轮不是 allocator 开关的七轮稳定确认。已有正式正确性与同配置版本性能证据，allocator 是否默认开启仍需同组重复对照支持，因此当前默认保持关闭。

make unit-test 覆盖独立 DFT 端到端结果、n_fft={1,2,4,8,32}、多个帧数、零/脉冲/随机输入、空 Mel 带、窄带、超过 Nyquist 的 f_max、非法参数以及单/多线程重复调用。tests/check_outputs.py 用 NumPy rfft 和独立稠密 Mel 公式覆盖 7 个小参数、当前题 4 组参数各自生成的 9 帧 PCM、heptagon 旧题 4 组 expected，共 15 个 case；默认 rtol=1e-3、atol=1e-4。此处缺少正式 PCM/expected 的陈述记录的是 R1 在 Prometheus 上的情况；正式题数据后来在 cluster 可用，R2 job 48/51 已用其进行验证，详见下文。短合成输入仍不替代正式输入回归。

别名回归新增三种 in-place 场景：输出小于输入、输出大于输入的多帧多线程/尾批，以及 n_fft=1 的零窗分支；期望值由独立直接 DFT 计算。最终 Prometheus 定向复验中，alias 单元测试在标量、AVX2、AVX2+HUGE_ALLOC 和 ASan+UBSan 配置均通过；allocator header 生命周期修复后的普通与 UBSan allocator-test 均通过。

新实现按帧块复用每 worker 的暂存空间。原实现对完整功率谱的理论数组大小约为 5.87 GiB（8192 点、192397 帧）和 11.74 GiB（16384 点、192333 帧）；这是理论内存占用，不是实测 RSS。

## Prometheus 验证结果

round1 定向构建、测试和基准在个人机器 Prometheus 上完成：Ubuntu x86_64、AMD Ryzen 9 8945HX、GCC 15.2.0。此机器不是集群；round1 记录时集群尚未准备，因此当时没有集群验证结果。Mac 只用于同步代码和取回结果，之前的本地编译仅为实现阶段历史，不作为验收证据。

最终修复后的验证结果：

- 新增 alias 回归在 SCALAR=1、AVX2、AVX2+HUGE_ALLOC 和 ASan+UBSan 单元测试中通过；测试覆盖输出缩小、输出扩大后的多帧/多线程尾批，以及 n_fft=1。
- allocator-test 的 HUGE_ALLOC 普通和 UBSan 构建通过，覆盖零/小/阈值/大块分配、对齐、SIZE_MAX/new_handler 和并发分配。placement new 显式开始 header 对象生命周期；未把 UBSan 当作生命周期正确性的证明。
- 最终 AVX2 与 AVX2+HUGE 两个构建各通过 15 个 NumPy case，共 30 个：7 个小参数、4 个当前题参数各自生成的 9 帧合成 PCM、4 个 heptagon 旧题 expected。当前题短合成输入最大绝对差为 6.0920149e-05；旧 expected 最大绝对差为 0.00390625；均满足 rtol=1e-3、atol=1e-4。
- 最终 8 核、1 核短基准的 original、reference、AVX2、AVX2+HUGE 输出逐 case 一致，最大绝对差为 0。
- 最终长输入比较三次运行均稳定，AVX2 和 AVX2+HUGE 对 reference 最大绝对差为 5.9604645e-08。
- 新 alias 测试格式化后重跑 AVX2 unit-test 与 clang-format --dry-run --Werror，均通过。Clang 构建、NumPy 全矩阵和 clang-analyzer 静态分析是在本次局部修复前做的，不声称本次修复后重新通过。

此前修复前的完整 75-case GCC/Clang NumPy 矩阵及 Clang/static 分析记录保存在 results/pre-review-fix/，只作为历史证据；最终该次局部修复阶段只报告上述 AVX2 与 AVX2+HUGE 30-case 复验。那一阶段 Prometheus 上的 data symlink 指向不可用的 `/cache/recuritment/melspec/data`，因此当时四组题目参数仅以 9 帧合成 PCM 验证，没有正式全量结果。正式题数据随后在 heptagon cluster 可用；R2 正式结果见“Round2 集群验收”小节。上述历史陈述不表示当前正式数据缺失。

### 128 帧基准

每组固定 seed=20261029，128 帧，运行 3 次。所有被测程序使用 GCC 15.2.0、-std=c++17 -O3 -pthread -mavx2 -mfma。8 核运行限定 taskset -c 0-7 且 MELSPEC_THREADS=8；1 核运行限定 taskset -c 0 且 MELSPEC_THREADS=1。表内为 median/min 毫秒；最后一列按 original median / AVX2 median 计算，不用 min 计算加速比。

| 核数 | case | original median/min | heptagon reference median/min | AVX2 median/min | AVX2+HUGE median/min | original/AVX2 median |
|---:|---|---:|---:|---:|---:|---:|
| 8 | default | 116.007/115.860 | 2.240/2.037 | 2.077/2.070 | 2.094/2.066 | 55.86x |
| 8 | small_mels | 45.072/43.877 | 1.981/1.911 | 1.716/1.565 | 2.246/2.067 | 26.26x |
| 8 | large_fft | 244.397/229.355 | 3.376/2.880 | 3.166/2.661 | 2.799/2.781 | 77.18x |
| 8 | narrow_band | 243.226/225.201 | 3.508/3.288 | 4.034/3.579 | 3.543/3.204 | 60.29x |
| 1 | default | 115.645/114.123 | 3.377/3.189 | 3.040/2.998 | 3.009/2.967 | 38.04x |
| 1 | small_mels | 44.592/44.428 | 3.079/2.956 | 2.949/2.887 | 2.923/2.851 | 15.12x |
| 1 | large_fft | 249.679/246.593 | 6.189/5.732 | 7.539/7.491 | 7.506/7.348 | 33.12x |
| 1 | narrow_band | 236.618/233.362 | 6.185/6.141 | 6.340/5.804 | 6.011/5.802 | 37.32x |

AVX2 相比 heptagon reference 在 8 个“核数 × case”组合中有 5 个 median 更快、3 个更慢；较慢项是 8 核 narrow_band、1 核 large_fft 和 1 核 narrow_band。短基准仅三次运行，median/min 会受 CPU 频率与缓存状态影响；这些数字不表示本次 correctness 修复造成了性能变化。短输入各项分配小于 1 MiB，未触发 HUGE allocator，因此不能由短基准归因 allocator 收益。HUGE_ALLOC 默认继续关闭。

### 长输入压力结果

长输入使用旧题 reference/data/audio.pcm（22,633,216 bytes）和当前 default 参数，得到 88,348 帧与 90,468,352 bytes 输出；每个构建运行 3 次。这是旧题 PCM 上的压力测试，不是当前题正式输入。

| 程序 | median/min 秒 | median peak RSS KiB | max peak RSS KiB | 相对 reference 最大绝对差 |
|---|---:|---:|---:|---:|
| heptagon reference | 0.307052/0.289793 | 364,532 | 364,820 | 0 |
| AVX2 | 0.376339/0.354286 | 359,732 | 359,792 | 5.9604645e-08 |
| AVX2+HUGE | 0.323461/0.279349 | 363,684 | 364,140 | 5.9604645e-08 |

此长输入 AVX2 median 比 reference 慢约 22.6%，AVX2+HUGE 慢约 5.3%；HUGE 的三次运行波动较大，不能据此宣称 allocator 提速。median peak RSS 分别比 reference 低 4,800 KiB 和 848 KiB。三个构建输出均稳定；AVX2 与 AVX2+HUGE 输出 hash 相同（dcc2671d4462cf45c2afeba2475e62c3d444cb80dbd10378fbc710798df1de52）。

R1/Prometheus 历史阶段没有逐步 ablation 或采样 profile；后续 R2 有 A/B 候选分项筛选，R3 使用 worker 内部采样记录 window/bit-reversal、FFT、power 和 Mel 阶段。算法迁移矩阵记录实现来源与代码差异，不代表每一项都经过单独性能归因。局部修复后非别名计算路径不复制输入，上一节以外的 benchmark 解释也不将性能变化归因于 alias 或 placement-new 修复。

### 复现命令与原始结果

远端 artifact 根目录为 /home/zige/workspace/melspec-optimization-20261029，本地同步目录为 /Users/zigeqiao/workspace/melspec-optimization-20261029。完整原始结果、日志和二进制 hash：

- results/validation-summary.md
- results/benchmark-8.json
- results/benchmark-1.json
- results/long-input.json
- results/build-manifest.txt

以下命令使用 tester 真实的构建目录与二进制路径。原版 current HEAD 与 heptagon reference 都在 Prometheus 上用 GCC 15.2 和相同基础 flags 构建。

    root=/home/zige/workspace/melspec-optimization-20261029
    build="$root/build-validation"
    export TMPDIR="$root/tmp"
    cd "$root/repo"

    # alias 单元测试：标量、AVX2、HUGE、ASan+UBSan；格式化后再跑 AVX2
    make clean && make SCALAR=1 unit-test
    make clean && make ARCH_FLAGS="-mavx2 -mfma" unit-test
    make clean && make HUGE_ALLOC=1 ARCH_FLAGS="-mavx2 -mfma" unit-test
    make clean && make CXXFLAGS="-std=c++17 -Wall -Wextra -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" unit-test
    make clean && make ARCH_FLAGS="-mavx2 -mfma" unit-test

    # allocator 普通与 UBSan 专项
    cd "$build/optimized_huge"
    make clean
    make CXX=g++ HUGE_ALLOC=1 ARCH_FLAGS="-mavx2 -mfma" allocator-test
    make CXXFLAGS="-std=c++17 -Wall -Wextra -O1 -g -fsanitize=undefined -fno-omit-frame-pointer" HUGE_ALLOC=1 allocator-test
    cd "$root/repo"

    # 最终 AVX2 与 AVX2+HUGE 各跑 15 个 NumPy case
    for name in optimized optimized_huge; do
      taskset -c 0-7 env MELSPEC_THREADS=8 "$root/venv/bin/python" tests/check_outputs.py \
        --program "$build/programs/$name" --reference-data "$root/reference/data"
    done

    # 格式检查；Clang/static 分析是此前版本证据，本轮没有重跑
    clang-format --dry-run --Werror src/melspec.cpp src/large_alloc.cpp tests/test_melspec.cpp tests/test_large_alloc.cpp

    # 8 核和 1 核基准；脚本 seed 默认 20261029，runs=3
    taskset -c 0-7 env MELSPEC_THREADS=8 "$root/venv/bin/python" tests/benchmark.py \
      --program "original=$build/programs/original" \
      --program "reference=$build/programs/reference" \
      --program "optimized=$build/programs/optimized" \
      --program "optimized_huge=$build/programs/optimized_huge" --runs 3 \
      --output "$root/results/benchmark-8.json"
    taskset -c 0 env MELSPEC_THREADS=1 "$root/venv/bin/python" tests/benchmark.py \
      --program "original=$build/programs/original" \
      --program "reference=$build/programs/reference" \
      --program "optimized=$build/programs/optimized" \
      --program "optimized_huge=$build/programs/optimized_huge" --runs 3 \
      --output "$root/results/benchmark-1.json"
    taskset -c 0-7 env MELSPEC_THREADS=8 "$root/venv/bin/python" "$root/run_long_benchmark.py"

GCC 原版与 reference 的构建分别使用：

    g++ -std=c++17 -O3 -pthread -mavx2 -mfma -I "$build/baseline/include" \
      -o "$build/programs/original" "$build/baseline/src/main.cpp" "$build/baseline/src/melspec.cpp"
    g++ -std=c++17 -O3 -pthread -mavx2 -mfma -I "$root/reference/include" \
      -o "$build/programs/reference" "$root/reference/src/main.cpp" "$root/reference/src/melspec.cpp"

最终重建的优化版构建变体命令为：

    cd "$build/optimized_avx2" && make clean && make CXX=g++ ARCH_FLAGS="-mavx2 -mfma"
    cd "$build/optimized_huge" && make clean && make CXX=g++ HUGE_ALLOC=1 ARCH_FLAGS="-mavx2 -mfma"
    cd "$build/optimized_scalar" && make clean && make CXX=g++ SCALAR=1 ARCH_FLAGS="-mavx2 -mfma"

original/reference 二进制复用首次 GCC 构建，final AVX2、HUGE、scalar 二进制已重建；来源和最终二进制 SHA256 见 build-manifest.txt。

## 来源与辅助说明

- 参考实现：/Users/zigeqiao/workspace/heptagon/melspec/src/melspec.cpp，commit 03d9542。
- GCC vector extension 文档：https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
- Linux Transparent Huge Pages 文档：https://kernel.org/doc/html/latest/admin-guide/mm/transhuge.html
- madvise(2)：https://www.man7.org/linux/man-pages/man2/madvise.2.html
- sched_getaffinity(2)：https://www.man7.org/linux/man-pages/man2/sched_getaffinity.2.html

本次实现与本报告由 AI 辅助生成，改动及结论应结合代码和后续测试结果审阅。

## 第二轮有限性能诊断与候选筛选

本轮只在 [src/melspec.cpp](/Users/zigeqiao/workspace/recruitment-2026-autumn/src/melspec.cpp) 保留候选 B：按 Mel bands 实际覆盖的连续 bin 范围，选择只重构低频侧 power，省去未访问的对称高频 power 写入；完整频带超出安全区间时仍走原来的完整重构。DC、Nyquist、中心 bin、空 band 与 alias 行为均保留。`tests/test_melspec.cpp` 增加 sr/4 两侧、高频 only、Nyquist 与超 Nyquist、全空和部分空 filter、中心 bin 命中，以及 8/17 帧尾批次的独立 DFT 覆盖。A1/A2 两个候选只留在 round2 artifact 中，没有进入产品源码。

### 冻结基线、机器与限制

修改前将 round1 的源码、头文件、测试、报告、dirty diff、构建环境和 5 个二进制快照保存到：

    /home/zige/workspace/melspec-optimization-20261029/round2/baseline

基线 `src/melspec.cpp` SHA256 为 `1dcce4b3354a1a96860695f6d97e663813d45317ad6abbbdbdcd5dd76b8d6e6e`，优化版二进制 SHA256 为 `b66db2e88bbaf3bad0776e0dc2eaa8c374f4d57f6662060d6a7b13ab8132312b`。本轮候选在 Prometheus 个人机器上用 GCC 15.2、`-std=c++17 -O3 -pthread -mavx2 -mfma` 构建，固定 affinity 为 `taskset -c 0`/`MELSPEC_THREADS=1` 或 `taskset -c 0-7`/`MELSPEC_THREADS=8`。Prometheus 不是集群。本轮筛选时 Prometheus 本地副本没有正式题目数据，因此 long 输入使用旧题 `reference/data/audio.pcm`（22,633,216 bytes），按当前 default 参数得到 88,348 帧；不把它称为正式数据。集群随后已上线，Main 确认 heptagon Slurm 计算节点和正式/cache 数据可用；最终正式数据验收由 tester 经 Slurm 计算节点执行，本轮没有在 login 节点运行计算。

内核 `perf_event_paranoid=4` 导致 perf 计数器不可用，本轮未再次尝试 perf，也没有采样 profile。报告的汇编观察来自静态 `objdump`，只说明代码生成，没有指令占比或时间归因。round1 源码反汇编中已看到 FFT 蝶形有 AVX/FMA、帧乘 hop 已被编译器移出热循环，scratch 每个 worker 只清零一次；A1 反汇编显示输入装载内循环不再按有效 lane 检查，窗系数改为 vector 乘法。A1+A2 汇编在完整 4-lane 路径生成 `vmovdqu` 宽 store，尾批次仍走标量 lane store。反汇编原件位于 `round2/results/objdump-round1-process-block.txt`、`objdump-A1-process-block.txt`、`objdump-A1A2-process-block.txt`。

### 阶段计时

新增 `tests/benchmark_compute.cpp`，通过 `-Dmain=melspec_cli_main` 编译未改的 round1 `src/main.cpp`，复用实际 `read_pcm`/`write_output`；分别链接 round1、reference、B 的 `melspec` object。每进程做一次全新输出的 compute。阶段定义为 read+normalize、compute、floatcast+write；compute 包含参数校验、Mel/twiddle/bit reverse 预计算、分配和线程启动，不等于纯 FFT。旧 PCM/current default 参数下各 1/8 核跑 3 次，以下列中是阶段 median 秒数和三阶段占比（占比以三阶段时间和为分母）：

| 程序 | affinity | read+normalize | compute | floatcast+write |
|---|---|---:|---:|---:|
| round1 | 1 核 | 0.0503（3.7%） | 1.2439（91.2%） | 0.0696（5.1%） |
| reference | 1 核 | 0.0347（2.6%） | 1.2497（93.6%） | 0.0512（3.8%） |
| B | 1 核 | 0.0512（3.8%） | 1.2318（91.0%） | 0.0707（5.2%） |
| round1 | 8 核 | 0.0511（14.3%） | 0.2351（65.9%） | 0.0706（19.8%） |
| reference | 8 核 | 0.0335（10.5%） | 0.2318（73.0%） | 0.0523（16.5%） |
| B | 8 核 | 0.0539（13.8%） | 0.2677（68.4%） | 0.0699（17.9%） |

阶段数据仅 3 次且用于解释端到端时间构成。B 的 default compute 阶段没有显示可确认的收益，因此本报告不声称 default 提速；端到端分解和全部单次数据在 `round2/results/compute-phases.jsonl` 与 `compute-phases-summary.json`。进程 wall time 与三个阶段之和不完全相等，包含启动、退出等开销。

### 交错 long benchmark 与候选决定

新增 `tests/benchmark_round2.py`：每个程序先做正确性预检和一次 warmup，再串行交错运行；输出文件读取、正确性比较和 SHA 计算都在计时区间之外。筛选期的 Prometheus JSON 保留当时实际运行顺序（两程序配对版本使用奇偶轮反序），不得视为后来的轮换脚本结果。最终集群验证后，脚本改为轮换程序起始位置、每个完整轮换周期反转顺序，并增加拒绝 NaN/Inf 输出的预检；cluster job 51 使用这一版本。所有 JSON 均记录每轮实测顺序、全部 samples、median/min、配对速度比/MAD、完整 argv、affinity、线程数、输入和二进制 SHA。`round2/results` 下的 A1/A1A2/A2/B JSON 是 Prometheus 筛选原始记录。

表内 baseline/candidate 都是各自这次交错运行的 median 秒数；配对比定义为每一轮 `baseline time / candidate time`，大于 1 表示 candidate 较快，MAD 是该配对比的中位绝对偏差。

| 候选 | 核数 | case | baseline median | candidate median | 配对比 median / MAD |
|---|---:|---|---:|---:|---:|
| A1 | 1 | default | 1.403688 | 1.421558 | 0.9869 / 0.0044 |
| A1 | 1 | narrow | 2.803077 | 2.880188 | 0.9836 / 0.0193 |
| A1 | 8 | default | 0.386034 | 0.401738 | 0.9393 / 0.0576 |
| A1 | 8 | narrow | 0.647719 | 0.608044 | 1.0365 / 0.0281 |
| A1+A2 | 1 | default | 1.421250 | 1.404351 | 1.0152 / 0.0112 |
| A1+A2 | 1 | narrow | 2.804650 | 2.844730 | 0.9966 / 0.0115 |
| A1+A2 | 8 | default | 0.388543 | 0.404938 | 0.9467 / 0.0382 |
| A1+A2 | 8 | narrow | 0.628386 | 0.623802 | 0.9971 / 0.0281 |
| A2 | 1 | default | 1.409457 | 1.446457 | 0.9830 / 0.0097 |
| A2 | 1 | narrow | 2.819668 | 2.839631 | 0.9880 / 0.0132 |
| A2 | 8 | default | 0.402919 | 0.405250 | 0.9797 / 0.0626 |
| A2 | 8 | narrow | 0.646294 | 0.657162 | 0.9747 / 0.0398 |
| B round 1 | 1 | default | 1.404698 | 1.375110 | 1.0114 / 0.0127 |
| B round 1 | 1 | narrow | 2.837754 | 2.642778 | 1.0749 / 0.0377 |
| B round 1 | 8 | default | 0.396822 | 0.390390 | 1.0391 / 0.0692 |
| B round 1 | 8 | narrow | 0.636418 | 0.596297 | 1.0699 / 0.0118 |
| B independent confirm | 8 | default | 0.399979 | 0.417908 | 0.9503 / 0.0685 |
| B independent confirm | 8 | narrow | 0.643895 | 0.625681 | 1.0270 / 0.0199 |

A1 是 frame pointer 预排并把窗乘移到 vector；A2 是完整 lane 以 `memcpy` 写宽向量、尾批次仍逐 lane 写。它们分别和组合版本都没有稳定、可重复的提升，所以只保留在 `round2/build` artifact。B 在 narrow long 的两轮 Prometheus 8 核 paired ratios 合并 median 为 1.0628、MAD 0.0296；两轮单独的 median ratio 分别是 1.0699 与 1.0270，方向均较快。Default 两轮方向不一致：首轮 median 约快 1.6%，确认轮约慢 4.5%，paired MAD 分别约 6.9% 与 6.8%；筛选阶段不能声称 default 提速，也尚不能据此称为重复回退。后续 Slurm 验收已在四组正式长输入上复测，结果见下节；没有出现 B 相对 round1 的长输入回退。当前不加 dispatch、阈值或其他候选。

round2 当前保留源码 SHA256 为 `0734a650a461aa271a2c4555d3b93e16b399ca78b68e03f2887d882077db7b29`，B 候选二进制 SHA256 为 `1d513bc1cb1cfa5b93b5702e16defaa9e860b22efddad60d2d4f3fff6ebc5b6c`。筛选版来源 hash 与二进制 hash：A1 `656d2bbf…` / `ace982d3…`，A1+A2 `70361c19…` / `e87a9065…`，A2 `4a4453ab…` / `f510b4ce…`；完整 64 位值及构建环境见 `round2/results/build-manifest.txt`。

### 本轮已做验证与复现位置

- B 的 AVX2 独立 DFT 单元测试通过，覆盖新增低频裁剪、回退路径、空 filter、Nyquist/超 Nyquist、中心 bin、1/2/4/8/32 FFT 长度、多帧尾批次；格式化后再次运行通过。日志在 `round2/results/B-unit-avx2.log`。
- `clang-format --dry-run --Werror` 对 B 源码、测试与阶段计时 harness 通过。
- Makefile `benchmark-compute` target 在 Prometheus 成功编译；仅有未改 `src/main.cpp` 的 `n_mels`/`n_frames` unused parameter 警告。
- 当时此段记录的是 round2 筛选完成、完整验收尚待执行的状态；最终 Slurm 验收结果及其限制见下节。集群构建和计算均由 Slurm 分配到计算节点运行，没有在 login 节点运行。

远端 round2 根目录为 `/home/zige/workspace/melspec-optimization-20261029/round2`，本地 artifact 镜像为 `/Users/zigeqiao/workspace/melspec-optimization-20261029/round2`。全部性能 JSON、harness 阶段数据、源码快照、二进制与 hash 在这些目录。代表性 8 核首轮与独立确认命令为：

    root=/home/zige/workspace/melspec-optimization-20261029
    export TMPDIR="$root/round2/work"
    taskset -c 0-7 env MELSPEC_THREADS=8 "$root/venv/bin/python" "$root/repo/tests/benchmark_round2.py" \
      --program "optimized=$root/round2/baseline/binaries/optimized" \
      --program "B=$root/round2/build/B/optimized_B" --baseline optimized \
      --pcm "$root/reference/data/audio.pcm" --suite current-long \
      --case default --case narrow_band --runs 7 --warmups 1 \
      --output "$root/round2/results/B-long-8.json" \
      --work-dir "$root/round2/work/B/8"
    taskset -c 0-7 env MELSPEC_THREADS=8 "$root/venv/bin/python" "$root/repo/tests/benchmark_round2.py" \
      --program "B=$root/round2/build/B/optimized_B" \
      --program "optimized=$root/round2/baseline/binaries/optimized" --baseline optimized \
      --pcm "$root/reference/data/audio.pcm" --suite current-long \
      --case default --case narrow_band --runs 7 --warmups 1 \
      --output "$root/round2/results/B-confirm-8.json" \
      --work-dir "$root/round2/work/B/confirm-8"

阶段 harness 数据和精确编译参数见 `round2/build/harness/build-commands.txt`、`round2/results/compute-phases.jsonl` 与 `compute-phases-summary.json`。本轮 Prometheus 筛选仍以旧题 PCM 为 stress input、非正式数据；heptagon 集群是在筛选后上线的。后续正式数据与 Slurm 计算节点的完整验收已完成，见下节。

### Slurm 正式数据验收与最终性能结果

Slurm job 48（验证）和 job 51（性能）均 `COMPLETED`、exit code 0。编译、正确性验证和计时都在 hepnode3 计算节点执行，不在登录节点或 Mac 上执行。节点为 Intel Xeon Gold 6338，job affinity 固定为 CPU 0–7（8 个核）；GCC 15.2、Python 3.14.4、NumPy 2.5.4。构建统一使用 `g++ -std=c++17 -Wall -Wextra -O3 -mavx2 -mfma -pthread`，按 AVX2、AVX2+HUGE_ALLOC、scalar 编译变体设置宏。源码 `src/melspec.cpp` SHA256 为 `0734a650a461aa271a2c4555d3b93e16b399ca78b68e03f2887d882077db7b29`；round1 AVX2 二进制为 `b66db2e88bbaf3bad0776e0dc2eaa8c374f4d57f6662060d6a7b13ab8132312b`，B AVX2 为 `0494a471444fbb1676329cafcfc1064fd664d829989bd505b5d1631a46dcfd84`，B AVX2+HUGE_ALLOC 为 `ed9dcd655f82d14fffe291db48edc97bec67cefaaecda02fd0a807c00cdaccd5`。完整来源、编译配置和六个 build SHA 见 `round2/cluster/results/build-manifest-48.txt`。

本轮使用题目数据 `/cache/recuritment/melspec/data`：约 47 MB 的 96 kHz PCM，以及总计约 611 MB 的四组 expected 输出。官方 `validate.py` 保持未修改；round1 AVX2 跑 1 次、B AVX2 与 B AVX2+HUGE_ALLOC 各跑 3 次，三者均为 4/4 PASS。官方表中的最大绝对误差分别为 default `3.12e-2`、small_mels `1.17e-2`、large_fft `1.25e-1`、narrow_band `6.25e-2`，均满足 `rtol=1e-3, atol=1e-4`。绝对误差最大值不能单独与 `atol` 比较判失败，判定使用相对与绝对容差组合。Librosa 不在计算节点环境中，因此仅跳过了可选的 Librosa 计时，不影响 expected 输出比对。

其余最终验证均通过：B 的 scalar、AVX2、AVX2+HUGE_ALLOC 单元测试及 AVX2 ASan+UBSan 通过；普通和 UBSan allocator 测试通过；阶段计时 harness smoke、benchmark 轮换平衡和 NaN/Inf 拒绝检查通过。B 的三个构建变体各自通过 11 个独立 NumPy 合成 case，共 33 项，最大绝对误差 `6.0920149e-5`。官方长输入中 B 与 round1 输出逐值完全相同；reference 与 B 的最大差为 `9.765625e-4`，处于官方容差内。短输入 128 帧的四 case 输出检查通过，但其启动和调度开销占比高，不据此下性能结论。计算节点没有 `clang-format`，所以本次远端格式检查标为跳过；Prometheus 前序格式检查 PASS 仍作为历史结果记录，不表示它在最终集群节点重跑过。

正式长输入基准每个程序 warmup 1 次、交错跑 7 次，固定 8 核和 `MELSPEC_THREADS=8`。表中时延单位为秒，`median/min` 是每个程序的独立统计；配对比按逐轮 `round1 / B` 计算，`MAD` 是该配对比的中位绝对偏差。配对比大于 1 表示 B 较快，它不是 median 时延下降百分比。

| Case | round1 AVX2 median/min | B AVX2 median/min | reference median/min | 配对比 median / MAD |
|---|---:|---:|---:|---:|
| default | 1.518437 / 1.514413 | 1.496907 / 1.477303 | 1.179305 / 1.157497 | 1.01646 / 0.00245 |
| small_mels | 0.947369 / 0.940351 | 0.928106 / 0.925525 | 0.773109 / 0.770620 | 1.02075 / 0.00294 |
| large_fft | 2.296257 / 2.288970 | 2.253945 / 2.246270 | 1.894124 / 1.874230 | 1.02003 / 0.00286 |
| narrow_band | 2.272057 / 2.252069 | 2.184240 / 2.172603 | 1.907424 / 1.845371 | 1.03391 / 0.01438 |

另有独立 AVX2+HUGE_ALLOC 组，不含 reference 程序：

| Case | round1 AVX2+HUGE median/min | B AVX2+HUGE median/min | 配对比 median / MAD |
|---|---:|---:|---:|
| default | 1.118030 / 1.115630 | 1.100721 / 1.097495 | 1.01735 / 0.00294 |
| small_mels | 0.765958 / 0.764423 | 0.743956 / 0.742331 | 1.03013 / 0.00909 |
| large_fft | 1.858546 / 1.850024 | 1.818301 / 1.814489 | 1.02181 / 0.00329 |
| narrow_band | 1.830343 / 1.821260 | 1.744566 / 1.736584 | 1.04917 / 0.00124 |

四个 case 中 B 相对 round1 的 paired ratio 均大于 1，且没有可重复长输入回退；B 因而保留。AVX2 组的差异约为 1.6–3.4%，HUGE_ALLOC 组约为 1.7–4.9%，其中 narrow_band 的方向与 Prometheus 筛选一致。除窄频带外，多数差异只有数个百分点；这些是特定机器、输入和七次采样的端到端结果，不外推为跨机器的普遍提速。AVX2 对比组中的 reference 自带 allocator，且其时延低于 round1 与 B；HUGE_ALLOC 比较组没有 reference，不能拿两组的数值拼接成 B 击败 reference 的对照结论。HUGE_ALLOC 仍是显式实验开关，不改为默认配置。

最终原始数据、样本运行顺序、argv、输入与程序 hash 位于本地镜像 `/Users/zigeqiao/workspace/melspec-optimization-20261029/round2/cluster/results/`，远端根为 `/home/qzg/workspace/melspec-optimization-20261029/round2/cluster/`。主要文件为 `benchmark-long-nohuge-51.json`、`benchmark-long-huge-51.json`、`benchmark-short-nohuge-51.json`、对应 `.log`，以及 `validation-48.log`、`performance-51.log`、`slurm-validation-48.log`、`slurm-performance-51.log`。Slurm 提交脚本和 worker 位于 `cluster/jobs/validation.sbatch`、`performance.sbatch`、`validation_worker.sh`、`performance_worker.sh`。复现时从 cluster 目录提交到计算节点，不要在 login 节点直接 make/validate：

    cd /home/qzg/workspace/melspec-optimization-20261029/round2/cluster
    validation_job=$(sbatch --parsable jobs/validation.sbatch)
    validation_job=${validation_job%%;*}
    performance_job=$(VALIDATION_JOB_ID="$validation_job" \
      sbatch --parsable --export="ALL,VALIDATION_JOB_ID=$validation_job" \
      --dependency="afterok:$validation_job" jobs/performance.sbatch)
    sacct -j "$validation_job,$performance_job" --format=JobID,State,ExitCode,NodeList

脚本将验证与计时绑定到 hepnode3 的 8 核 Slurm allocation；job 51 的具体解释器为 `cluster/venv-round2-48/bin/python3`，NumPy 2.5.4。worker 还会从题目目录读取真实 PCM 与 expected 文件。正式输出的逐轮参数和二进制路径以 JSON `environment`、`programs` 字段为准。

### Prometheus 虚拟环境路径修复记录

Prometheus 上的 venv 从临时目录搬入 workspace 后，激活脚本、pip/NumPy 命令入口 shebang 和 `pyvenv.cfg` 仍引用旧路径，导致 shell 激活后 PATH 错误。已在原 venv 内精确修正 9 个文本文件、10 处旧前缀，先备份后替换并保留文件权限；备份、清单和 diff 在 `/home/zige/workspace/melspec-optimization-20261029/round2/environment-repair/20261030T034810Z/`。fresh Bash 检查确认 `source` 后 `python`、`python3`、`pip` 和 NumPy 2.5.4 均指向 workspace venv，`f2py`、`numpy-config` 可运行，`deactivate` 后 PATH 逐字节恢复。csh/fish 未安装，所以只审阅了相应脚本内容，未进行实际交互式 shell 测试。此前 benchmark 使用绝对 venv 解释器路径，因此该环境修复不改变既有验证或性能数据。用户可用以下命令重新激活修复后的环境：

    source /home/zige/workspace/melspec-optimization-20261029/venv/bin/activate
    hash -r

### Job 51 后 benchmark 输出新鲜度检查

Job 51 之后的审查发现 benchmark 在每次启动子进程前未清除同路径的旧输出，存在命令成功退出却误读残留文件的风险。已修正 `tests/benchmark_round2.py`：预检、warmup 和计时运行前都删除目标输出，计时运行的删除操作位于 `perf_counter()` 之前。Job 51 实际使用的旧工具版本已归档（SHA256 `cfe05c35991cee43e995136ede6c62033e8ca833c94228c2c13da26f4d2c0176`）；修正版本 SHA256 为 `342dbda9860a42f2066a6dc03e9752d32e9c09ddf25ceb48f56e0a07c7169ef7`，已同步到 Prometheus 和 cluster 副本。

独立 guard 验收在 Prometheus CPU 8–15 上用修复后的绝对 venv 解释器和 NumPy 2.5.4 完成。预检阶段带残留文件但子程序不写、warmup 只在预检写、计时调用只在预检写这三种失败用例，均以 `FileNotFoundError` 正确失败，且没有生成成功 JSON 或保留旧输出。真实 round1/B 的 current-short default 单次 smoke 输出完全一致（max absolute difference 为 0）；此 smoke 只验证输出，不用于性能结论。Job 51 使用全新 work directory 且各程序均写出新结果，因此正式性能矩阵有效、未重采样；原始结果仍可依据归档的 job51 工具版本追溯复现。

修复前后脚本、fixture、完整日志与 smoke JSON 保存在 `/Users/zigeqiao/workspace/melspec-optimization-20261029/round2/benchmark-output-guard/20261030T054254Z/`；其中 `README.md`、`logs/run-output-guards.log` 和 `smoke/actual-short-smoke.json` 记录了工具哈希、负例及真实程序验证证据。

## 第三轮 FFT 与跨帧宽度候选

本轮以 round2 的 B 版本为基线，在算法候选前先做正式输入阶段计时和低频采样诊断。所有 R3 快照、诊断源码、候选 build、脚本、日志和 JSON 均保存在 `/Users/zigeqiao/workspace/melspec-optimization-20261029/round3/`；远端对应 `/home/qzg/workspace/melspec-optimization-20261029/round3/`。round2 `src/melspec.cpp` 基线 hash 为 `0734a650a461aa271a2c4555d3b93e16b399ca78b68e03f2887d882077db7b29`，基线 B AVX2 二进制 hash 为 `0494a471444fbb1676329cafcfc1064fd664d829989bd505b5d1631a46dcfd84`。快照前的当前工作树、HEAD、dirty diff、round2 源码/reference、六个二进制和正式 JSON 见 `round3/baseline/`。

构建和计时在 hepnode2 的 Slurm compute allocation 上完成：Intel Xeon Gold 6338，GCC 15.2，CPU affinity 0–7，Python 3.14.4、NumPy 2.5.4；AVX2、FMA、AVX-512F 均由该节点的 CPU flags 确认。期间 hepnode3 有其他 8 核 allocation；两节点共享 `/home` 和输入文件系统，因此 read/write 阶段可能受共享存储缓存和 I/O 影响。没有在 Mac 或 login 节点编译、测试或 profile。

### 阶段计时与 FFT 门槛

round3 复用了 `tests/benchmark_compute.cpp` 和真实 `read_pcm`/`write_output`，对正式 96 kHz PCM 的四组参数做 1 次 warmup、3 次交错运行；compute 包含 Mel/twiddle/bit-reversal 准备、分配与线程启动，不代表纯 FFT。输出文件写完取 size 后即删除。以下为阶段中位数秒数，来自 hepnode2 job 117：

| Case | Build | read+normalize | compute | floatcast+write |
|---|---|---:|---:|---:|
| default | B | 0.184729 | 0.832058 | 0.450455 |
| default | B + HUGE_ALLOC | 0.074959 | 0.659562 | 0.357800 |
| default | reference | 0.071916 | 0.689781 | 0.361826 |
| small_mels | B | 0.182088 | 0.675678 | 0.057930 |
| small_mels | B + HUGE_ALLOC | 0.073448 | 0.634559 | 0.034935 |
| small_mels | reference | 0.070789 | 0.664299 | 0.035201 |
| large_fft | B | 0.183355 | 1.558560 | 0.491772 |
| large_fft | B + HUGE_ALLOC | 0.072745 | 1.377480 | 0.359029 |
| large_fft | reference | 0.072030 | 1.438130 | 0.360461 |
| narrow_band | B | 0.182916 | 1.481330 | 0.493329 |
| narrow_band | B + HUGE_ALLOC | 0.074123 | 1.304600 | 0.358314 |
| narrow_band | reference | 0.072626 | 1.410350 | 0.359480 |

阶段计时的单次 read/write 有明显共享 I/O 和缓存差异，不用它们单独下 allocator 结论。未插桩的 B 与 B+HUGE_ALLOC+reference CLI 对照另在 job 121 同组运行，数据列在下一表。阶段 JSON 与逐轮样本见 `round3/results/compute-phases-baseline-job117.json`。

profile-only 源码副本位于 `round3/profile/melspec_profile.cpp`；每个 worker 每 64 个跨帧 group 采一组，分别计时 window+bit-reversal 输入装载、FFT、power 重构和 Mel 累加。每个 case 的四段 worker 累计采样时间比例为：

| Case | 采样 groups | window+bit-reversal | FFT | power | Mel | 插桩 compute 中位数差异 |
|---|---:|---:|---:|---:|---:|---:|
| default | 752 | 26.29% | 63.52% | 5.65% | 4.55% | -0.18% |
| small_mels | 752 | 26.29% | 63.97% | 5.70% | 4.06% | +0.19% |
| large_fft | 752 | 26.81% | 64.08% | 5.28% | 3.86% | +0.25% |
| narrow_band | 752 | 28.50% | 67.82% | 1.97% | 1.78% | +0.15% |

`FFT` 比例的分母是被采样 group 的四个 kernel 阶段之和，worker 时间跨线程累加；它不是整次运行的 wall-time 分解，也不是 perf 采样占比。插桩和未插桩的 compute 中位数差均小于 0.3%，因此 1/64 采样率保留。四 case 的 FFT 采样占比都高于计划中的 25% gate，允许继续实施有限 F/V 候选。setup wall time（含校验、Mel/FFT 表、window、bit-reversal、输出准备，线程启动和 scratch 另计）中位数依次为 200.764、50.898、201.151、201.275 ms；worker scratch 建立时间为各 worker 累加值，字段保存在 `profile-overhead-job117.json` 原始数据中。

### 同组 baseline CLI 对照

round2 B AVX2、B AVX2+HUGE_ALLOC 和原生 reference 在 hepnode2 job 121 同一 8 核 allocation 中交错运行，warmup 1 次、每个 case 3 次。paired ratio 定义为逐轮 `B time / program time`；此处只作本轮三次基线观察，不将 3 次结果当作 allocator 的稳定收益。所有程序都通过 benchmark 输出一致性检查：B 与 B+HUGE_ALLOC 输出逐值相同，reference 与 B 的最大差为 `9.765625e-4`。

| Case | B median/min | B + HUGE_ALLOC median/min | reference median/min | B/HUGE ratio median / MAD | B/reference ratio median / MAD |
|---|---:|---:|---:|---:|---:|
| default | 1.522846 / 1.497195 | 1.101496 / 1.100614 | 1.128602 / 1.125116 | 1.3824 / 0.0194 | 1.3535 / 0.0136 |
| small_mels | 0.944329 / 0.939248 | 0.749278 / 0.748372 | 0.776934 / 0.772703 | 1.2618 / 0.0012 | 1.2191 / 0.0030 |
| large_fft | 2.268900 / 2.268609 | 1.825374 / 1.817904 | 1.879563 / 1.870126 | 1.2460 / 0.0019 | 1.2099 / 0.0030 |
| narrow_band | 2.194882 / 2.181176 | 1.743171 / 1.740372 | 1.848250 / 1.848099 | 1.2612 / 0.0005 | 1.1801 / 0.0066 |

JSON `round3/results/cli-baseline-8-job118.json` 记录程序 binary hash、输入 digest、完整 argv、真实交错顺序、原始 samples 和 consistency；JSON 自身不含 Slurm job id。实际运行是实施阶段提交记录中的 job 121，Main 只读核对的 `sacct` 状态为 hepnode2 `COMPLETED 0:0`（15:39:32–15:41:30 UTC）；JSON 文件名中的 `job118` 是工作目录标签，不能当作 Slurm allocation 编号。阶段 JSON、profile 和候选 build 来自 job 117，最终状态为 `FAILED 141:0`，原因和已落盘子步骤见下文。HUGE_ALLOC 的正式七轮验证留给 tester，不据三轮基线表直接推荐默认 allocator。

### F/V 候选与定向验证

仅新增两个显式、默认关闭的构建开关：`FUSED_FFT=1` 将相邻 radix-2 stage 成对融合，使用原 twiddle 表和原复数乘法表达式，stage 数为奇数时留最后一级走原 radix-2 路径；`AVX512=1` 在 `SCALAR=0` 且编译器支持 AVX-512F 时将跨帧宽度设为 8 lane。SCALAR 分支优先；AVX-512 请求若缺少 `-mavx512f`，Makefile 给出配置错误。已有 SIMD packing、tail handling、power/Mel 路径及 alias 备份未变。产品默认仍是 B，未加入运行时 dispatch。

候选 flags 统一为 `g++ -std=c++17 -Wall -Wextra -O3 -pthread -mavx2 -mfma`；V/FV 额外加 `-mavx512f`。round3 实测构建及二进制 SHA256：

| Candidate | FUSED_FFT | AVX512 | Binary SHA256 |
|---|---:|---:|---|
| B | 0 | 0 | `0494a471444fbb1676329cafcfc1064fd664d829989bd505b5d1631a46dcfd84` |
| F | 1 | 0 | `a9c6a8b0cc5155fecdb05bcf1e1470b9eb3c969e0fa42d86edb0dc5ef0e4bb54` |
| V | 0 | 1 | `133f3a64fa607a6d4fdf2fb6b2d8e446012f4a6aed34653d4f135447a81baf8a` |
| FV | 1 | 1 | `02790eb66e23d687e3ca1e7bbb489d81fc4e6f6561caa71e5d24ba759cc12a1b` |
| scalar + F | 1 | 0 (`SCALAR=1`) | `f32cb31e1c7e29e4927962c01b79850325ab9f24ebd04ccc6b219238ef15ce17` |

五个变体在 hepnode2 均成功构建，并各自通过 `make unit-test`；新增独立 DFT 覆盖 n_fft 16/64/128 与 6/10/15/16/18 帧，同时保留原小 FFT、alias、空带、高频/中心 bin、线程测试。候选没有跑正式 expected 全量或长输入候选性能矩阵，binary 也未据此作排名，交由 tester 对 B/F/V/FV 做计划内筛选。

`objdump` 静态检查保存在 `round3/results/objdump-{B,F,V,FV}-job117.txt`。worker lambda 中 B/F 分别包含 YMM 运算，V/FV 分别生成 125/179 条使用 ZMM 寄存器的指令；扫描该 lambda 未发现 YMM/ZMM 到栈的直接 spill store。F/FV fused loop 对四个 complex 结果发出八个向量结果 store，避免在源码上逐级写入并重读中间 stage；这是静态代码结构证据，不是内存流量或性能测量。

job 117 的阶段 JSON、profile、候选构建日志、对象文件和反汇编均在 `round3/`。该 Slurm job 最终状态为 `FAILED 141:0`：所有阶段数据、四个候选构建、五组单元测试与反汇编都已在终止前完成；随后用于列出符号的 `grep | head` 在启用 `pipefail` 的交互 shell 中触发 SIGPIPE，导致 shell 以 141 退出。应将具体结果按已保存的 JSON、build manifest 和日志审阅，不把整个 job 标为 PASS。candidate build 源码 SHA256 为 `712867d887e9af3b214806079ef72aa2220adddafeec08017359772d1271cce5`，Makefile 为 `05adf9ef8e634d564ea8dea97102cb61b6bf4b5c4902d0a312203738806deab4`，测试为 `dddbddf257a9dea43e36f788a946ac7361a6c4e442476603e125e7ea467c1c18`；完整命令与候选哈希见 `round3/results/candidate-build-manifest-job117.txt`。下一步由 tester 独立完成 B/F/V/FV 正式四 case correctness 与性能筛选，再决定是否保留或启用任何候选。

### R3 正式验收计划与停止条件

截至本报告补全，Round3 tester 还没有提交 B/F/V/FV 的正式验收作业；workspace 下 `round3/tester/jobs/`、`results/`、`work/` 仅为后续工作目录。当前可确认的 R3 结果限于 job 117 已完成的构建、定向单测、低频采样和静态反汇编；job 121 是三轮 B/BHuge/reference 阶段基线，不是候选验收，也不是七轮 allocator 确认。以下是本次有限候选实验的内部筛选/停止规则，不是官方评分标准：

1. 先验证输出正确性，再做同节点、同 affinity、同输入、交错顺序的性能比较。配对速度比定义为 baseline 耗时 / candidate 耗时，值大于 1 表示 candidate 较快。
2. B/F/V/FV 各跑四组正式长输入 3 次筛选；按四组 paired ratio 的几何平均排序，候选需 GM ≥ 1.02 且每组 ratio ≥ 0.98，最多保留两个候选进入确认。
3. 入围候选与 B 做 7 次同组确认；需 GM ≥ 1.03、至少三组改善、没有一组 paired median < 0.98，且声称的收益超过相应 2×MAD。所有七轮样本与每轮程序顺序都应保留。
4. 入围者再做四组单核初筛；出现超过 3% 的可疑回退时增加到 7 轮确认。HUGE_ALLOC 必须在同组、同 allocator 条件下比较，不能把不同实验组的绝对时间拼在一起下结论。
5. 若没有候选达到稳定门槛，继续使用 B 默认实现；最多进行一次由 profile 证据指导的有限 replan，不连续增加未经证实的优化分支。

R3 采样显示 FFT 阶段占被采样 worker kernel 时间约 63–68%，但这是 worker 累计计时比例，不是整次程序的 wall-time 占比。作为一般 Amdahl 解释，若一个占总运行时间 5% 的工作被完全消除，理论总加速上限约为 `1/(1-0.05)=1.053x`；这只是公式示例，不是本项目的测量或性能承诺。当前报告包含算法来源、数学等价、正确性边界、正式数据回归、端到端性能、性能诊断、失败候选和复现命令，足以说明已完成的工作与证据；R3 是否带来可复现提升仍须等待上列独立验收。

### 当前已验证配置的 Slurm 复现

下面命令复现已由 R2 job 48/51 验证的 B + HUGE_ALLOC 二进制，不会重建程序。正式数据位于 `/cache/recuritment/melspec/data`；从 repo 目录调用 validator，使其原有数据路径解析保持不变。Python 用绝对 venv 解释器，不依赖激活脚本。编译、测试、验证与计时均应在计算节点运行，不在 login 节点运行：

    ssh heptagon
    srun -p hepnodes --nodelist=hepnode3 -N 1 -n 1 -c 8 \
      --time=00:20:00 --cpu-bind=cores --pty bash

    cr=/home/qzg/workspace/melspec-optimization-20261029/round2/cluster
    export MELSPEC_THREADS=8
    export TMPDIR="$cr/work/manual"
    mkdir -p "$TMPDIR"
    cd "$cr/repo"
    "$cr/venv-round2-48/bin/python3" validate.py \
      "$cr/build/validation-48/B_huge/melspec" --runs 3

    time "$cr/build/validation-48/B_huge/melspec" \
      /cache/recuritment/melspec/data/audio.pcm "$TMPDIR/manual.bin" \
      --sr 96000 --n_fft 8192 --hop_length 128 --n_mels 256 \
      --f_min 0 --f_max 22050
    rm -f "$TMPDIR/manual.bin"
    exit

validator 的四组 `4/4 PASS` 和 B_huge 的 hash 已在上方记录；`validate.py` 的计时输出中 Librosa 一栏因计算节点未安装 Librosa 可为 N/A，这不影响 expected 输出比较。若需从源重新构建，应在同类 compute allocation 中执行 `make clean && make CXX=g++ HUGE_ALLOC=1 ARCH_FLAGS="-mavx2 -mfma"`，不要在 login 节点运行；该构建命令不重现 R3 F/V，后两者分别要显式设置 `FUSED_FFT=1` 或 `AVX512=1`，且 AVX-512F 只能运行于支持它的 CPU。

### R3 证据索引与解释边界

- Round1/Prometheus 历史数据：本地 workspace `results/` 与 `round2/results/`，远端 Prometheus workspace 对应目录；其中旧题 PCM 长输入不代表当前题正式数据。
- Round2 正式集群结果：本地 `/Users/zigeqiao/workspace/melspec-optimization-20261029/round2/cluster/results/`，远端 `/home/qzg/workspace/melspec-optimization-20261029/round2/cluster/results/`；job 48/51 的日志、JSON 和 `build-manifest-48.txt` 证明源码与二进制版本。
- Round3：本地 `/Users/zigeqiao/workspace/melspec-optimization-20261029/round3/`，远端 `/home/qzg/workspace/melspec-optimization-20261029/round3/`。关键文件为 `baseline/README.md`、`results/compute-phases-baseline-job117.json`、`results/profile-overhead-job117.json`、`results/diagnostic-summary-job117.txt`、`results/cli-baseline-8-job118.json`、`results/candidate-build-manifest-job117.txt`、`results/objdump-{B,F,V,FV}-job117.txt` 和 `candidate-source/`。

job 117 最终 `FAILED 141:0` 的原因是末尾诊断 shell 管道的 SIGPIPE；保留的独立 JSON、单测/build 日志和反汇编可用于核对具体步骤，但整体作业不能记为 PASS。job 121 的 Slurm 编号依据实施阶段提交记录及 Main 只读 `sacct` 结果，不来自 `cli-baseline-8-job118.json` 的 JSON 字段；该 JSON 文件名只保留了其 work-directory 标签。profile 百分比来自低频 worker 阶段计时，`objdump` 来自静态汇编；二者都不应描述为 perf CPU profile 或单项加速实测。

## 第四轮：用满整机 64 核 + MKL FFT

### 动机：前三轮遗漏的两个大方向

1. **核数**：前三轮的所有正式测量都固定 8 核（`taskset -c 0-7` / Slurm `-c 8`）。heptagon 计算节点 hepnode1–3 实际为 2× Xeon Gold 6338（64 物理核、无超线程、4 个 NUMA 节点、L2 1.25 MiB/核、支持 AVX-512），`/tmp` 是 tmpfs。题目允许使用整机资源，compute 阶段又是按帧完全独立的并行任务，扩展到 64 核是收益最大的一步。
2. **FFT 本身**：R3 采样显示 FFT 占 worker kernel 时间 63–68%。手写实现是跨帧 SIMD 的 radix-2：每一级都要把整块（`n_fft/2` 个复数 × lane 数 × 16 B，default 为 256–512 KiB）从 L2 完整扫一遍，共 log2(n_fft/2) 级，带宽受限；再加 bit-reversal 的随机写。题目明确允许 FFTW/cuFFT 等科学计算库，集群 spack 里没有 FFTW，但有 Intel oneMKL 2026.0（DFTI 接口），在 Intel CPU 上有针对 AVX-512 的混合基、cache-oblivious 实现。

### 新的计算结构（`MKL=1`，默认在检测到 MKL 时启用）

- 每个 worker 一个 MKL DFTI descriptor：`DFTI_DOUBLE`、`DFTI_REAL`、原地、`DFTI_COMPLEX_COMPLEX` 存储、`THREAD_LIMIT=1`；链接**静态** `mkl_sequential`，不引入 MKL 自己的线程池，也不需要运行时 `LD_LIBRARY_PATH`。
- 帧按 16 帧一个 tile 连续分给 worker。tile 内逐帧：加窗拷贝到 `n_fft+2` 的缓冲 → 原地 r2c → 只对 Mel 实际覆盖的 bin 区间 `[active_first, active_end)` 求 `|X|²`，并以 `[bin][frame]` 的转置布局写入 power tile。
- Mel 累加对一个 tile 的 16 帧做向量化（每个 bin 的 power 行是 16 个连续 double，编译器生成 AVX-512 FMA），每个滤波器内仍按 bin 升序求和，数学上与原实现逐项一致；结果行连续写入 `output[m * n_frames + frame0 ...]`。
- 全程保持 double 精度，输出仍由未改动的 `main.cpp` 转成 float32。
- 线程启动/异常传播抽成 `run_workers()`，手写 FFT 路径（`MKL=0`）保留为回退，并复用同一函数。

### 构建默认值变化

- `ARCH_FLAGS ?= -march=native`（构建在 Slurm 计算节点上进行，与运行节点同型号）。
- Linux 上 `HUGE_ALLOC` 默认改为 1：R3 job 121 同组三轮显示 B+HUGE_ALLOC 相对 B 的 paired ratio 为 1.25–1.38，且 read/write 阶段（`main.cpp` 中的大 vector 首次触页）同样受益；这是一个“未改 main.cpp 也能加速 main.cpp 内存分配”的手段。
- `MKLROOT` 默认指向集群 spack 中的 oneMKL；找不到时 `MKL` 自动为 0，回退到 R2/R3 的手写 FFT。

### R4 测量环境与一个测量陷阱

- 节点：hepnode2，`sbatch -N1 -c 64 --exclusive`，GCC 15.2，所有计时都在计算节点上跑；`validate.py` 与 `src/main.cpp` 均未修改。
- **陷阱**：第一次 MKL 验证（job 139）中我把 `TMPDIR` 设到了 `/home` 下的工作目录，`validate.py` 的 `tempfile.mkstemp()` 于是把 200–400 MB 输出写到了共享文件系统上，default 测得 0.58–0.61 s；同一程序写 tmpfs `/tmp` 时为 ~0.38 s。阶段计时也印证了这一点：`floatcast+write` 在 `/home` 上约 0.37 s，在 `/tmp` 上约 0.14 s。官方 validate 默认使用 `/tmp`，因此之后所有 R4 测量都让输出落在 `/tmp`。R3 job 117 的 B 行 write 阶段偏慢（0.45–0.49 s）很可能也有同样的原因。

### R4 基线：同一份 R2 B+HUGE 代码，从 8 核换到 64 核（job 137，validate 5 次取最小）

| Case | 8 线程 (s) | 64 线程 (s) |
|---|---:|---:|
| default | 0.887 | 0.380 |
| small_mels | 0.758 | 0.232 |
| large_fft | 1.609 | 0.467 |
| narrow_band | 1.530 | 0.466 |

只改线程数就有 2.3–3.4 倍提升，但远小于 8 倍：在 64 核时，`main.cpp` 中不可修改的串行读写已经与 compute 同一量级。

### R4 profile：64 核时时间去了哪里

阶段计时（job 145，default，输出写 `/tmp`）：read+normalize 0.07 s、compute 0.16 s（其中 `output.resize()` 单线程清零 394 MB 占 0.027 s）、floatcast+write 0.14 s，进程总墙钟时间约 0.38 s。也就是说，即便把 compute 降为 0，默认 case 也只能到约 0.24 s；剩下可优化的主要是 kernel 本身。

`MELSPEC_TRACE=1` 打开的 worker 内部计时（job 147/148）：

| 观测 | 数据 | 结论 |
|---|---|---|
| 线程启动 + MKL descriptor commit | 64 个 worker 在 ≤16 ms 内全部开始计算 | 不是瓶颈 |
| worker 完成时间 | default 0.083–0.145 s，large_fft 0.19–0.32 s | 连续静态分块**负载不均**，最慢的 worker 比最快的慢 50% 以上 |
| 线程扩展（default compute） | 1/8/16/32/64 线程：5.00/0.66/0.35/0.20/0.13–0.16 s | 32 线程前接近线性，64 线程时开始受频率、带宽和不均衡影响 |
| 单线程分阶段（default，每帧） | 加窗 3.8 µs、FFT 14.6 µs、power 3.8 µs、Mel 3.5 µs | FFT 占约 57%；power 的 `[bin][frame]` 转置写每次都落在新 cache line，代价接近加窗 |
| 单线程分阶段（narrow_band，每帧） | 加窗 7.6 µs、FFT 36.3 µs、power 3.2 µs、Mel 2.5 µs | 16384 点 FFT 的工作集变大，FFT 占约 73% |
| MKL 与手写 FFT（单线程 default compute） | 4.94 s 与 5.06 s | 在这台机器上，手写跨帧 AVX2 radix-2 与 MKL 单帧 r2c 基本持平；单换 FFT 库不是突破口 |

### R4 优化 1：动态 tile 调度 + 行布局 power + tile 化 Mel 输出（job 150）

根据上面的 profile 做了三处改动（`compute_with_mkl`）：

1. **动态调度**：不再把连续帧静态均分给 worker，而是用一个 `std::atomic<size_t>` 计数器，每次领取 2 个 tile（32 帧）。领取粒度足够大，原子操作开销可忽略（全程约 6000 次 `fetch_add`）；同时快线程自动多干，NUMA 远端、降频或被系统打断的线程少干。
2. **power 改成每帧连续的一行**：`|X[k]|²` 直接写到连续数组，去掉了之前 `[bin][frame]` 转置写的 cache line 浪费；Mel 对每个滤波器做连续 bin 的点积，求和顺序仍是 bin 升序。
3. **Mel 结果先放进 `[mel][16 帧]` 的 tile（256×16×8 B = 32 KiB，放得进 L1/L2）**，tile 完成后每个 Mel 行连续写 16 个 double 到输出，避免每帧在输出中跨 256 行写入。

| 指标（hepnode2，64 线程） | 改动前 (job 148) | 改动后 (job 150) |
|---|---:|---:|
| default kernel | 0.145 s | **0.083 s** |
| default worker 完成时间范围 | 0.084–0.145 s | 0.080–0.083 s |
| narrow_band kernel | 0.225 s | **0.164 s** |
| 单线程 default power 阶段 | 0.73 s | 0.16 s |

不均衡基本消除（最慢 worker 只比最快慢 4%）。改动后 64 线程 worker 0 的分解为加窗 0.010 s、FFT 0.048 s、power 0.003 s、Mel+写出 0.011 s，**FFT 占约 2/3**，成为唯一的大头。

### R4 优化 2：把输出清零与 FFT 计算重叠

`std::vector<double>::resize()` 必须对 394 MB 输出做值初始化，标准库会在调用线程上单线程 memset，64 线程时实测 0.025–0.028 s，约占 compute 的 1/4。这一步无法省略（接口固定为 `std::vector<double>&`，`main.cpp` 不可改），但可以与计算重叠：

- 先在 `compute_melspectrogram` 中完成参数校验、Mel 权重、Hann 窗等准备工作，再把 `output.resize()` 包成 `prepare_output` 回调交给 `compute_with_mkl`。
- worker 0（即调用线程）先执行 resize，完成后通过 `std::atomic<int>` 以 release 语义发布 `kReady`；其他 63 个 worker 一启动就创建 DFTI plan、领取 tile 计算，输出还没准备好时，把 32 KiB 的 tile 结果暂存在私有 `stash` 中，每领取一次任务检查一次状态（acquire），一旦就绪先 flush stash 再直接写。
- 若 resize 抛出（如 `bad_alloc`），worker 0 发布 `kFailed` 后重新抛出，其他 worker 看到后直接退出，`run_workers` join 后把异常传给调用方，不会死等。
- alias（输入与输出是同一个 vector）路径在任何 worker 启动前已复制输入，worker 只读副本，resize 不会影响它们。

### R4 实验开关：分配器触页线程数

`HUGE_ALLOC` 分配器对 ≥1 MiB 的分配做 2 MiB 对齐 mmap + `MADV_HUGEPAGE`，然后多线程预先触页。原来硬编码最多 16 个触页线程；在 64 核节点上新增编译期参数 `MELSPEC_TOUCH_THREADS`（默认仍为 16），用来测试 32/64 线程能否进一步缩短 `main.cpp` 中 `y`、`out_f32` 等大 vector 的分配时间。


### R4 原始朴素实现基线（job 175–178，单核，写 `/tmp`）

为了给最终加速比一个分母，用仓库初始提交 `be4db00` 的 `src/melspec.cpp`（朴素 O(N²) DFT、单线程）按原 Makefile 编译，四个 case 各占一个 1 CPU 作业。原版 default 单 case 就要半小时以上，`validate.py` 的 5 次重复不可行，这里只计一次墙钟时间（包含读写）。narrow_band 跑了近一小时仍未结束，考虑到集群是共享的且很拥挤，主动取消，之后不再运行原始实现；要对照参考语义时直接阅读 `melspec.py` 和原始源码，后续加速比以已测的三个 case 和 R2/R3 的优化版本为准：

| case | 原始实现墙钟 |
|---|---:|
| default | 2058 s |
| small_mels | 562 s |
| large_fft | 4432 s |
| narrow_band | 未测（运行 58 分钟后主动取消） |

### R4 一个只在真实 MKL 头文件下出现的编译错误

本地没有 MKL，语法检查用的是自己写的 `mkl_dfti.h` 桩。集群第一次编译 r7（job 185）失败：MKL 把 `DftiCreateDescriptor` 定义成**5 参数宏**（按精度和维数分派到 `_s_1d`/`_d_1d` 等函数），而我在实参里写了 `std::is_same<Real, float>::value ? ...`，模板实参里的逗号把宏参数切成了 6 个。修复方法是先算出 `const DFTI_CONFIG_VALUE precision` 再传给宏。教训：桩头文件只能保证"函数签名对得上"，宏展开相关的问题必须在真实环境编译才能发现，所以之后每次改动都先在集群上 `make unit-test`。

### R4 变体对比（job 196，hepnode，48 核非独占）

集群拥挤，独占 64 核的作业长时间排不上，这一组用 `-c 48` 非独占节点，所以绝对时间比独占 64 核略慢，适合横向比较。变体：

- `d1`：当前代码默认（double FFT，逐帧 DFTI）
- `d4`：double FFT，`MELSPEC_MKL_BATCH=4`，一次 DFTI 调用做 4 帧
- `f1` / `f4`：`MELSPEC_FFT_FLOAT=1`，加窗与 FFT 用 float，power 转回 double 后做 Mel（f4 为 4 帧批量）
- `d1t64`：`d1` + `MELSPEC_TOUCH_THREADS=64`

**kernel 时间（`MELSPEC_TRACE=1`，3 次）**

| 变体 | default kernel | narrow_band kernel | default w0：加窗/FFT/power/Mel |
|---|---|---|---|
| d1 | 0.109 / 0.111 / 0.112 | 0.220 / 0.208 / 0.214 | 0.011 / 0.048 / 0.003 / 0.011 |
| d4 | 0.105 / 0.117 / 0.109 | 0.310 / 0.216 / 0.209 | 0.011 / 0.043 / 0.003 / 0.010 |
| f1 | 0.069 / 0.066 / 0.074 | 0.121 / 0.127 / 0.124 | 0.004 / 0.013 / 0.001 / 0.008 |
| f4 | 0.073 / 0.074 / 0.070 | 0.119 / 0.127 / 0.117 | 0.005 / 0.016 / 0.001 / 0.010 |
| d1t64 | 0.112 / 0.113 / 0.116 | 0.218 / 0.221 / 0.252 | 0.009 / 0.047 / 0.003 / 0.010 |

所有变体的 worker 完成时间极差都在 1–8 ms 以内，动态调度在 48 线程下同样有效。`stashed` 显示部分 worker 在输出就绪前暂存了最多 80–190 个 tile，说明"清零与计算重叠"确实在起作用：清零期间别的 worker 没有闲着。

**官方 `validate.py --runs 5`（写 `/tmp`，取最小值，单位 s）**

| 变体 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| R2 B_huge | 0.400 | 0.246 | 0.490 | 0.476 |
| d1 | **0.336** | **0.216** | **0.465** | **0.433** |
| f1 | 0.305 | 0.173 | 0.348 | 0.350 |
| d1t64 | 0.345 | 0.226 | 0.455 | 0.443 |

**误差余量（`tests/tolerance_margin.py`，usage = max|diff|/(atol+rtol·|exp|)，必须 < 1）**

| 变体 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| R2 B_huge | 0.0005 | 0.0008 | 0.0008 | 0.0006 |
| d1 | 0.0005 | 0.0008 | 0.0008 | 0.0006 |
| f1 | 0.0250 | 0.0050 | 0.0152 | 0.0117 |

结论：

1. **d1 与 R2 的输出逐位一致**（误差统计完全相同），说明新的 MKL + tile 路径在 double 下没有引入可见的数值差异；同条件下比 R2 快 6–16%。
2. **批量 FFT（d4）没有稳定收益**：default FFT 阶段略快（0.043 vs 0.048），narrow_band 反而有一次 0.31 s 的离群值，三次中位数与 d1 持平。原因是批量后每次调用的工作集变成 4×16384×8 B ≈ 0.5 MB，挤占了 L2。默认保持 `BATCH=1`。
3. **触页线程 64（d1t64）没有收益**，在噪声范围内甚至略慢；默认保持 16。
4. **float FFT 快很多**：FFT 阶段从 0.048 s 降到 0.013 s（AVX-512 一次 16 个 float），large_fft/narrow_band 端到端快约 25%。代价是误差余量用量从 0.08% 升到最多 2.5%，仍有约 40 倍余量，四组 validate 全部 PASS。但仓库自带的 `test_melspec` 单元测试是对 double 实现的精确比较，float 版本在第 6 位有效数字后不同，所以单元测试判为失败。

关于 float：README 要求"数学等价"，`include/melspec.h` 注释写明 "All calculations in double"。float FFT 的计算公式不变，但中间精度降低，严格说不是同一个计算。因此**默认构建保持 double**，float 只作为编译开关 `CPPFLAGS=-DMELSPEC_FFT_FLOAT=1` 提供，并在这里给出完整的速度与误差数据，由评审决定是否接受。

### R4 优化 3：把 `write_output` 的缺页移进 compute（在 prometheus 上先验证）

集群排队时间长，从这一步起先在个人机器 prometheus（AMD Ryzen 9 8945HX，16 核 32 线程，AVX-512，单 NUMA，`/tmp` 为 tmpfs）上验证想法，再提交集群。prometheus 上没有 MKL，用 PyPI 的 `mkl-static`/`mkl-include` 2024.2 wheel 解出静态库，通过 `MKLROOT=` 指向它，无需改 Makefile。prometheus 是 AMD CPU，MKL 的 FFT 性能不代表集群，所以 prometheus 只用来验证正确性和与 FFT 无关的部分。

**main.cpp 的 I/O 拆分。** 64 核时 `main.cpp` 的读（约 0.07 s）和写（约 0.14–0.15 s）已占端到端时间的 2/3，而这两部分源码不能改。为了知道分配器还能影响哪一部分，写了 `tests/io_breakdown.cpp`：把 `read_pcm`/`write_output` 原样复制，在每条语句之间计时（prometheus，3 次）：

| 步骤 | 无 HUGE_ALLOC | HUGE_ALLOC（触页 1/4/16 线程） |
|---|---:|---:|
| `vector<int16_t> buf(n)` 分配 + 清零（49 MB） | 0.015–0.018 | 0.007–0.010 |
| `f.read` | 0.007–0.011 | 0.005–0.007 |
| `y.resize(n)`（98 MB） | 0.066–0.068 | 0.033–0.039 |
| int16 → double 转换 | 0.021–0.022 | 0.018–0.021 |
| `vector<float> out_f32(n)`（197 MB） | 0.056–0.062 | 0.032–0.037 |
| double → float 转换 | 0.036–0.042 | 0.034–0.040 |
| `ofstream::write`（写 tmpfs） | 0.058–0.063 | 0.060–0.067 |

触页线程数 1/4/16 的结果在噪声范围内，这和集群上 `d1t64` 无收益一致。再把 197 MB 的 `out_f32` 单独拆开测：HUGE_ALLOC 下 `operator new`（mmap + 缺页 + 内核清零）约 0.017 s，之后 libstdc++ 值初始化的 memset 约 0.018 s，再写一遍只要 0.015–0.020 s。所以 `out_f32` 分配有一半时间是**缺页和内核清零**，而这一步发生在 compute 结束之后的串行路径上。

**做法。** compute 开始时已经知道输出元素数，而 `main.cpp` 的 `write_output` 一定会紧接着分配一个同样元素数的 `vector<float>`。于是：

- `large_alloc.cpp` 新增 `melspec_reserve_allocation(bytes)`：在调用线程上（单线程触页，不再额外起线程，避免与 64 个 worker 争核）mmap + `MADV_HUGEPAGE` + 预触页一块内存，挂到一个 `std::atomic<AllocationHeader*>` 槽位上。
- `operator new` 对 ≥1 MiB 的请求先看槽位：请求不超过容量、且不小于容量一半（避免把 197 MB 的块给一个 1 MB 的请求）时，用 CAS 取走直接返回，不再缺页。块的头部与普通 mmap 块相同，`operator delete` 照常 munmap。
- `compute_with_mkl` 中 worker 0 在 resize 输出并发布 `kReady` 之后调用它。此时其他 worker 正在算 tile，动态调度会让 worker 0 少领几个 tile 来吸收这段时间。worker 0 就是调用线程，也就是之后执行转换循环的线程，所以这些页面首次触碰就落在该线程所在的 NUMA 节点上。
- `melspec.cpp` 通过**弱符号**引用这个函数：单元测试等不链接 `large_alloc.cpp` 的目标中它的地址为空，调用会被跳过；`HUGE_ALLOC=0` 时则完全不编译这段代码。

这不改变任何计算和输出，只是把一次必然发生的分配提前到并行阶段完成。它依赖"`main.cpp` 下一步会分配输出大小的 float 数组"这一程序结构事实，与输入数据无关；如果预测落空（比如被别的程序复用），代价只是一块未使用的映射，进程退出时由内核回收。

prometheus 上 `benchmark_compute`（5 次，default）：写阶段从 0.138–0.172 s 变为 0.133–0.139 s，compute 不变（0.28–0.32 s）。`validate --runs 5` 交替跑两轮四组全部 PASS，prometheus 上 compute 占比大、笔记本温度波动明显，端到端差异在噪声内。集群对比见 r8（job 205）。

**集群结果（hepnode3，48 核非独占）。** `benchmark_compute` 交替 4 次：default 写阶段 d1 0.139–0.160 s → r1 0.142–0.146 s；narrow_band 0.152–0.157 s → 0.141–0.152 s；compute 无变化（worker 0 多做的预触页被动态调度吸收）。官方 `validate.py --runs 5`，d1 与 r1 交替两轮（job 206；job 205 的 validate 因测试目录漏建 `data` 链接而失败，与程序无关，已重跑）：

| 版本 / 轮次 | default | small_mels | large_fft | narrow_band |
|---|---:|---:|---:|---:|
| d1 第 1 轮 | 0.335 | 0.217 | 0.468 | 0.437 |
| r1 第 1 轮 | **0.323** | 0.207 | **0.447** | 0.439 |
| d1 第 2 轮 | 0.328 | 0.215 | 0.453 | 0.435 |
| r1 第 2 轮 | **0.316** | 0.215 | **0.443** | 0.435 |

四组全部 PASS，误差与 d1 完全相同。收益约 10–15 ms（default/large_fft，约 3–4%），narrow_band、small_mels 持平。收益小于 prometheus 上的拆分估计，原因是集群上 16 线程触页本来就比较快，被移走的只有缺页部分。这项改动没有副作用，保留为默认。

### R4 最终验收（job 207，hepnode3，`--exclusive` 独占 64 核）

同一节点、同一作业内交替运行 R2 B_huge、R4 默认（r1，double）和 R4 float 开关（r1f），`validate.py --runs 5`，写 `/tmp`，各两轮，单位 s：

| 版本 | 轮次 | default | small_mels | large_fft | narrow_band |
|---|---|---:|---:|---:|---:|
| R2 B_huge | 1 / 2 | 0.370 / 0.373 | 0.226 / 0.232 | 0.450 / 0.461 | 0.442 / 0.449 |
| **R4 默认（double）** | 1 / 2 | **0.307 / 0.307** | **0.188 / 0.191** | **0.406 / 0.403** | **0.389 / 0.384** |
| R4 float 开关 | 1 / 2 | 0.280 / 0.272 | 0.159 / 0.160 | 0.317 / 0.322 | 0.305 / 0.300 |

误差余量（max_usage，越小越好，<1 为通过）：R4 默认 0.0005/0.0008/0.0008/0.0006，与 R2 完全一致；float 开关 0.0250/0.0050/0.0152/0.0117。

R4 默认的 64 线程阶段时间（`benchmark_compute`，3 次）：default read 0.067–0.074、compute 0.088–0.109、write 0.133–0.147；narrow_band read 0.070–0.073、compute 0.167–0.197、write 0.141–0.144。

### R4 总结

**端到端加速（validate 时间，取两轮较小值）**

| case | 原始实现（单次墙钟） | R2 B_huge 8 核（job 137） | R2 B_huge 64 核 | **R4 默认 64 核** | 相对原始 | 相对 R2 8 核 | 相对 R2 64 核 |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 2058 s | 0.887 | 0.370 | **0.307** | ≈6700× | 2.9× | 1.21× |
| small_mels | 562 s | 0.758 | 0.226 | **0.188** | ≈3000× | 4.0× | 1.20× |
| large_fft | 4432 s | 1.609 | 0.450 | **0.403** | ≈11000× | 4.0× | 1.12× |
| narrow_band | 未测 | 1.530 | 0.442 | **0.384** | — | 4.0× | 1.15× |

原始实现只跑了一次、包含进程读写，与 validate 的 5 次取最小不完全同口径，倍数只说明数量级。

**R4 各步骤的贡献（主要依据）**

1. 从 8 核换到 64 核（同一份 R2 代码）：default 0.887 → 0.380 s。前三轮的测量都局限在 8 核，这是遗漏最大的一项。
2. 动态 tile 调度 + 行布局 power + tile 化输出：64 线程 kernel 0.145 → 0.083 s，worker 不均衡从 >50% 降到约 4%。
3. 输出清零与计算重叠：从串行路径上移走约 27 ms 的单线程 memset（其他 worker 暂存 tile）。
4. 输出 float 副本预触页：default/large_fft 再快 10–15 ms。
5. MKL 本身：单线程与手写跨帧 SIMD FFT 基本持平（4.94 vs 5.06 s）。它的价值在于让每个 worker 独立处理单帧，从而可以做细粒度动态调度，而不在于 FFT 本身更快。

**试过但没有采用的**

| 尝试 | 结果 | 原因 |
|---|---|---|
| MKL 批量 FFT（4 帧/次） | 无稳定收益，narrow_band 有离群慢值 | 4×16384 点工作集挤占 L2 |
| 触页线程 16 → 64 | 无收益 | 触页受内核缺页路径限制，线程数不是瓶颈 |
| float32 FFT | 快 10–25%，误差余量用量最高 2.5% | 不满足"double 计算"的接口约定和自带单元测试的精确比较；仅保留为开关 |
| GPU、滑动 DFT、裁剪 FFT | 未实现 | GPU 的启动与传输开销和 0.1 s 级 compute 相当；滑动 DFT 在 hop=128、N=8192 时不划算且数值不稳定；输出裁剪 FFT 理论收益 <10% 且要放弃 MKL |

**剩余瓶颈。** 64 核时 R4 默认 default 的 0.307 s 中，`main.cpp` 的读（约 0.07 s）和写（约 0.14 s）占约 2/3，而 compute 只有约 0.09 s。读写中剩下的是 `ifstream::read`、libstdc++ 值初始化 memset、单线程类型转换和写 tmpfs 的内核拷贝，这些都在不可修改的 `main.cpp` 里，分配器能影响的部分（缺页、大页）已经处理。compute 内部 FFT 约占 2/3，继续缩短只能靠降低精度（float 开关）或更多核。

**复现。** 在集群上 `sbatch slurm.job`（`-c 64 --exclusive`，内容为 `make clean; make test`）。默认构建自动使用 spack 中的 oneMKL 2026.0；也可 `make MKLROOT=...` 指定其他 MKL，`make MKL=0` 回到手写 FFT 路径。实验脚本和全部日志在集群 `~/workspace/melspec-optimization-20261029/round4/`（`jobs/`、`results/`）。

**AI 使用说明。** 第四轮的代码、实验设计、集群作业和本节文字由 AI（Claude）在用户指导下完成；用户提供了集群与个人测试机，并给出了约束（不改 `main.cpp`/`validate.py`/`data`、不要跑原始实现等）。所有数字都来自上文列出的作业日志。

## 第五轮：继续压缩 compute 与串行路径

### R5 想法 1：把 Hann 窗移到频域，FFT 直接读输入

周期 Hann 窗可以写成三个复指数之和：

$$w[n] = \tfrac12 - \tfrac14 e^{2\pi i n/N} - \tfrac14 e^{-2\pi i n/N}$$

所以加窗帧的 DFT 精确等于未加窗 DFT 的三点组合：

$$Y[k] = \tfrac12 X[k] - \tfrac14\left(X[k-1] + X[k+1]\right)$$

其中 $X[-1] = \overline{X[1]}$、$X[N/2+1] = \overline{X[N/2-1]}$（实信号的共轭对称）。这是恒等变形，不是近似；`melspec.py` 用的是 `get_window('hann', n_fft, fftbins=True)`，代码里的窗是 `0.5 * (1 - cos(2πi/N))`，分母是 N，正是周期形式，恒等式成立。

实现上的收益：

- 原来每帧要先把 N 个样本乘窗写进 FFT 缓冲区（in-place 变换会覆盖输入，这一份拷贝省不掉）。现在 DFTI 设成 `DFTI_NOT_INPLACE`，直接从 `samples + frame * hop` 读，结果写到两端各多留一个复数的频谱缓冲区。
- 两端的保护元素用共轭对称填好后，三点组合和 `|·|²` 在一个无分支循环里完成，而且只算 Mel 会用到的活跃 bin（default 46%、narrow_band 17%），不是全部 N/2+1 个。
- 编译开关 `MELSPEC_FREQ_WINDOW`（默认 1），只在 double、非批量路径生效（float 路径必须先把 double 输入转成 float，拷贝仍然存在）。

第一版实现有一个控制流错误：在频域分支末尾写了 `continue`，跳过了同一 tile 循环里随后的 `store_tile`，输出全为 0。自带单元测试立即报出 `actual=0.000000 expected=0.139371`；改为 `if constexpr … else` 后通过。单元测试以 2e-9 的相对容差对照直接 DFT 参考，说明频域加窗的数值误差远小于 validate 容差。

prometheus 上（32 线程，3 次）：误差余量与原来逐位相同（0.0005/0.0008/0.0008/0.0006）；narrow_band kernel 0.62 → 0.54 s（−14%），default 持平（0.29–0.30 s）。在 AMD 上 MKL 的 out-of-place 变换略慢，抵消了 default 省下的加窗时间；Xeon 上的结果见集群作业。

### R5 想法 2：输出数组在主线程所在 NUMA 节点上触页

`output`（394 MB）原来由分配器的 16 个辅助线程预触页，这些线程被内核分散到 4 个 NUMA 节点上；之后 `main.cpp` 的单线程转换循环读整个数组，大部分是跨节点访问。另外，worker 0 执行 `output.resize()` 时，这 16 个触页线程会和 63 个计算 worker 抢核。

改法是复用上一轮的预留机制：worker 0 在 `prepare_output()` 之前先 `melspec_reserve_allocation(输出字节数)`（单线程触页），`resize` 的分配直接取走这块内存。页面因此都在主线程所在节点上，也不再额外起线程。worker 0 多花的时间仍然由动态调度吸收。编译开关 `MELSPEC_LOCAL_OUTPUT`（默认 1）。prometheus（单 NUMA）上 default kernel 快 2–6%，来源只可能是去掉了线程争抢；NUMA 方面的效果要在集群上看。

### R5 一次误判：plan 创建并不慢

64 线程 trace 里 `planned` 的最大值达到 0.07–0.11 s，与 kernel 时间同量级，一度怀疑 `DftiCommitDescriptor` 内部加锁导致串行化。写了一个微基准：8192/16384 点单次 commit 0.4–3.6 ms，32 个线程并发 commit 全部在 2.3 ms 内完成，用 `DftiCopyDescriptor` 也一样。原因其实是 worker 0：它先执行 `resize` 和预留，最后才建 plan，所以 `planned` 的最大值是 worker 0 的，这正是有意安排的重叠，不是问题。记录下来，避免以后再误判。

### R5 集群对比（job 213，hepnode3，`--exclusive` 64 核）

四个版本：`r1` = 上一轮提交（HEAD）；`fw1` = 频域加窗；`lo0` = 只做 NUMA 本地输出；`lo` = 两者都做。单元测试全部 PASS，`lo` 的误差余量与 r1 逐位相同（频域加窗不改变误差统计）。

**compute 阶段（`benchmark_compute`，3 次，s）**

| case | r1 | fw1 | lo0 | lo |
|---|---|---|---|---|
| default | 0.089 / 0.089 / 0.110 | **0.086 / 0.078 / 0.080** | 0.113 / 0.123 / 0.114 | 0.112 / 0.112 / 0.116 |
| large_fft | 0.205 / 0.200 / 0.194 | **0.153 / 0.153 / 0.154** | 0.196 / 0.191 / 0.195 | 0.162 / 0.160 / 0.163 |
| narrow_band | 0.172 / 0.176 / 0.181 | **0.136 / 0.134 / 0.142** | 0.170 / 0.176 / 0.190 | 0.144 / 0.141 / 0.151 |

**`validate.py --runs 5`，两轮（s）**

| 版本 | default | small_mels | large_fft | narrow_band |
|---|---|---|---|---|
| r1 | 0.293 / 0.287 | 0.194 / 0.193 | 0.386 / 0.394 | 0.390 / 0.380 |
| **fw1** | 0.285 / 0.294 | **0.183 / 0.180** | **0.362 / 0.362** | **0.350 / 0.348** |
| lo | 0.318 / 0.317 | 0.182 / 0.182 | 0.364 / 0.362 | 0.351 / 0.343 |
| lo0 | 0.331 / 0.318 | 0.194 / 0.190 | 0.402 / 0.396 | 0.377 / 0.382 |

结论：

1. **频域加窗采用。** 在 Xeon 上 out-of-place 变换没有变慢，省掉的加窗拷贝直接体现为 compute 缩短：large_fft −24%，narrow_band −22%，default −10%；端到端 large_fft/narrow_band 快 7–9%，small_mels 快 6%，default 在噪声内（compute 只占它的 1/3）。
2. **NUMA 本地输出放弃。** trace 显示 `lo`/`lo0` 的 default 中 worker 完成时间是 `0.084..0.111`，最晚的正是 worker 0：它要单线程触页 394 MB、做 memset，再给 float 副本触页 197 MB，这段串行工作超过了 default 整个 kernel 的时长，动态调度已经无法吸收。写阶段也没有因为页面在本地而变快（0.14–0.15 s 不变），说明单线程转换循环不受远端访问限制。代码已删除。

这一轮说明：把串行工作"藏"到 worker 0 上是有预算的，预算就是其他 worker 的计算时间。default 的 kernel 只有约 0.08 s，worker 0 上的 resize（约 0.027 s）加 float 预留已经接近上限。

### R5 系统层面的检查（job 218，hepnode3，独占 64 核）

**THP 与缺页。** 节点的 `transparent_hugepage/enabled` 与 `defrag` 都是 `madvise`，所以分配器的 `MADV_HUGEPAGE` 生效，缺页时允许直接整理内存拿到大页。`/usr/bin/time` 统计整个进程（default）约 3.9–5.3 万次 minor fault，大头来自 <1 MiB 的小块分配和 MKL 内部缓冲；如果 700 MB 的大数组没拿到大页，仅它们就会有约 17 万次缺页。

**集群上的 I/O 拆分（`tests/io_breakdown.cpp`，HUGE_ALLOC，3 次取典型值，s）**

| 触页线程 | buf 分配 | 读文件 | y 分配 | int16→double | out_f32 分配 | double→float | 写 tmpfs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.009–0.011 | 0.026–0.030 | 0.041–0.042 | 0.020 | 0.032 | 0.041–0.042 | 0.072 |
| 4 | 0.006–0.007 | 0.026–0.029 | 0.027 | 0.020–0.021 | 0.018–0.019 | 0.045–0.046 | 0.074–0.075 |
| 16 | 0.005–0.006 | 0.024–0.028 | 0.024–0.025 | 0.021–0.022 | 0.014–0.015 | 0.049–0.051 | 0.076–0.078 |

16 线程触页对分配步骤最好，维持默认。分配器能影响的两个分配步骤已接近下限（out_f32 在真实程序里已由 R4 预留机制移出串行路径）。剩下的读文件、两次类型转换和写 tmpfs 都是 `main.cpp` 里的串行代码或内核拷贝，合计约 0.19 s，是当前 default 0.28 s 的主体。

注意 double→float 转换在 16 线程触页时反而慢约 8 ms：测试程序里 `data` 被 16 个线程分散触页到 4 个 NUMA 节点，单线程读它要跨节点。但 R5 的 `lo` 实验已经表明，把真实程序的 `output` 放到本地节点，写阶段并没有变快，所以这一点不值得用 worker 0 的串行时间去换。

**MKL 指令集。** `MKL_ENABLE_INSTRUCTIONS=AVX512` 与默认的 compute 时间相同（default 0.080–0.085 s、large_fft 0.153–0.159 s），`=AVX2` 时慢 10–15%（0.090–0.094 / 0.172–0.181 s）。MKL 在这颗 Ice Lake 上已经自动选了 AVX-512 代码路径，无需改动。

**线程绑核（新开关 `MELSPEC_PIN`，默认 0）。** worker i 绑定到进程可用的第 i 个 CPU（节点上 0–15、16–31、32–47、48–63 各为一个 NUMA 节点），计算结束后恢复调用线程原来的 mask（RAII，异常时也恢复）。第一版结果：

| | default compute | large_fft compute | validate default | validate large_fft | validate narrow_band |
|---|---|---|---|---|---|
| 不绑核 | 0.078–0.083 | 0.153–0.155 | 0.282 / 0.283 | 0.363 / 0.368 | 0.354 / 0.353 |
| 绑核 v1 | 0.116–0.120 | 0.149–0.151 | 0.324 / 0.330 | 0.352 / 0.358 | 0.342 / 0.347 |

large_fft/narrow_band 快约 3%，default 却变慢 40%，trace 中最后完成的是 worker 0。原因是 worker 0 先把自己绑到 CPU 0，然后才执行 `output.resize()`，分配器起的 16 个触页辅助线程继承了这个 affinity，全部挤在 CPU 0 上，相当于串行触页 394 MB。改为 worker 0 在准备完输出、做完预留之后再绑核（v2），见下一个作业。

**绑核 v2（job 219）。** worker 0 改为准备完输出后再绑核。compute（3 次）：small_mels 0.073–0.078 vs 不绑核 0.075–0.080；default 0.101–0.107 vs 0.080–0.106；large_fft 0.145–0.158 vs 0.151–0.166；narrow_band 0.140 vs 0.139–0.144。validate 三轮互有胜负（default 0.311/0.287/0.306 vs 0.286/0.303/0.306；large_fft 0.372/0.366/0.366 vs 0.364/0.377/0.371）。v1 在 large_fft 上的 3% 没有复现，绑核不采用，代码已删除。

### R5 优化：float 副本的预留交给 worker 1

job 219 的 trace 还暴露了一个和绑核无关的问题：不绑核的版本 default 也有一次 compute 达到 0.106 s，最后完成的仍是 worker 0。worker 0 上串行排着两件事：`output.resize()`（394 MB 触页 + memset）和 R4 加入的 float 副本预留（197 MB 单线程触页），两者加起来接近 default 整个 kernel 的时长（约 0.08 s），动态调度吸收不了，所以偶尔出现 25 ms 左右的拖尾。

这两件事互不依赖，且 R5 `lo` 实验已经说明 float 副本放在哪个 NUMA 节点无关紧要，所以把预留改由 worker 1 在启动时执行，与 worker 0 的 resize 并行（只有 1 个 worker 时仍由 worker 0 在 resize 之前做）。output 的字节数是 float 副本的两倍，大于预留块容量，`operator new` 的取用条件保证 resize 不会误拿这块内存。prometheus 上单元测试通过，`MELSPEC_THREADS=1` 与 32 线程的输出逐字节相同。集群对比见 job 221。

**结果（job 221，独占 64 核）。** compute 中的拖尾确实消失了（default 0.080–0.082 s，worker 完成时间极差 ≤2 ms），但端到端 validate 的 default 三轮都变慢：0.303/0.306/0.306 s，不移的版本是 0.289/0.295/0.281 s；其余三组在噪声内。差别来自写阶段：float 副本的页面改由 worker 1 首次触碰后，落在 worker 1 所在的 NUMA 节点上，而 `main.cpp` 的转换循环要在主线程上逐个**写**这个数组。R5 `lo` 实验只说明了远端**读**（`output`）影响不大；远端写的代价明显更高。所以 float 副本必须由主线程（worker 0）预触页，这一改动撤回。偶尔出现的 worker 0 拖尾（约 1/3 的运行多 20 ms 左右）是这个选择的代价，平均来看仍然划算。

### R5 实验：分配器把大块内存放到调用线程的 NUMA 节点（job 222，hepnode2）

上一项说明远端写很贵。`main.cpp` 读路径上的 `y`（98 MB，被 memset 和 int16→double 循环写）和 `buf`（49 MB，被内核读文件写）同样由分散在 4 个节点上的 16 个辅助线程首次触页，主线程写它们大多是远端写。于是在分配器里，触页前用 `getcpu` 取调用线程的节点，再对整个映射调用 `mbind(MPOL_PREFERRED)`（裸 syscall，不依赖 libnuma；该节点内存不足时自动回退）。16 个线程仍然并行触页，但页面都落在主线程所在节点。

| `io_breakdown`（3 次，s） | buf 分配 | 读文件 | y 分配 | int16→double | out_f32 分配 | double→float | 写 tmpfs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 不绑定节点 | 0.006–0.008 | 0.028–0.029 | 0.024 | 0.021–0.022 | 0.014–0.015 | 0.049–0.050 | 0.076–0.078 |
| 偏好本地节点 | 0.005–0.006 | 0.026–0.030 | 0.026–0.027 | 0.019 | 0.015–0.018 | 0.041–0.042 | 0.072–0.073 |

单看 I/O 路径收益是真实的：转换和写文件合计快约 15 ms；分配慢 2–3 ms，因为所有缺页都压在一个节点的内存控制器上。但真实程序的 validate 三轮（default 0.292/0.295/0.283 vs 0.297/0.284/0.281，narrow_band 0.350/0.342/0.354 vs 0.353/0.360/0.350，其余两组同样互有胜负）没有稳定差异。原因：真实程序里 `out_f32` 本来就由 worker 0 预留，已经是本地的；而 `output` 放到本地节点后，64 个 worker 里有 3/4 的写变成远端写，抵消了主线程读 `output` 的收益。不采用，代码撤回。

### R5 小结

本轮采用的只有频域加窗（large_fft/narrow_band 端到端快 7–9%）。另外五项尝试都没有采用，各自有明确的原因：NUMA 本地输出、绑核 v1/v2、float 副本预留交给 worker 1、分配器偏好本地节点；MKL 指令集已经是最优的 AVX-512，无需改动。目前 default 约 0.28–0.29 s 中，`main.cpp` 的读（约 0.07 s）和写（约 0.14 s）占约 3/4，compute 约 0.08 s。在不改 `main.cpp`、保持 double 精度的前提下，继续优化的空间只剩噪声量级。唯一还有明显收益的是 float FFT（compute 再省约 40%），但它涉及精度取舍，保留为开关。

### R5 收尾：两件串行准备工作分给两个 worker（job 224，hepnode2）

综合上面两项实验：float 副本必须由主线程首次触碰（它会被主线程写满）；`output` 的页面放在哪里无关紧要（反正由 16 个辅助线程并行触页）。于是互换分工：**worker 1 执行 `output.resize()`**（其他 worker 照旧暂存 tile 直到输出就绪），**worker 0（主线程）同时预留 float 副本**。两件串行工作并行进行，任何一件都不会超过最短 kernel 的时长；float 副本仍是主线程本地的。只有 1 个 worker 时退回原来的顺序。`MELSPEC_THREADS=1/2/32` 的输出逐字节相同。

| compute（3 次，s） | small_mels | default | large_fft | narrow_band |
|---|---|---|---|---|
| 原分工（fw1） | 0.078 / 0.079 / 0.078 | 0.082 / 0.083 / 0.083 | 0.155 / 0.159 / 0.153 | 0.138 / 0.144 / 0.143 |
| 新分工（sw） | 0.077 / 0.076 / 0.075 | 0.081 / 0.078 / 0.081 | 0.154 / 0.155 / 0.156 | 0.142 / 0.138 / 0.142 |

validate 三轮（default 0.287/0.291/0.290 vs 0.282/0.294/0.294；narrow_band 0.354/0.347/0.362 vs 0.355/0.350/0.360）在噪声内。本次两个版本都没有出现 worker 0 拖尾，所以端到端收益无法从噪声中区分出来；但它去掉了 job 207/219/221 中 worker 0 拖尾的成因，compute 也略好，且没有代价，因此采用。

### R5 优化：Mel 点积 4 帧一组累加（job 229，hepnode3，独占 64 核）

在不开 `-ffast-math` 的情况下，GCC 不会重排浮点归约，所以每个 Mel 频带的 `sum += p[k] * w[k]` 是一条串行 FMA 依赖链，每一步都要等上一次加法完成（约 4 个周期）。改为同时对 4 帧做同一个频带：先算出 4 帧的 power 行，再用 4 个独立累加器 `s0..s3` 共享同一个权重 `w[k]`。每一帧仍然按 bin 升序单独求和，**结果与原来逐位相同**（集群上四组参数用 `cmp` 对比 HEAD 构建的输出：全部 bitwise identical），只是 4 条依赖链可以在流水线中交错执行。tile 末尾不足 4 帧时，缺的行指向最后一个有效行，结果不写回。

| 指标（3 次） | 原版 | 4 帧一组 |
|---|---|---|
| default，worker 0 的 Mel 阶段 | 0.0070–0.0077 s | 0.0050–0.0058 s |
| large_fft，worker 0 的 Mel 阶段 | 0.0159–0.0180 s | 0.0119–0.0125 s |
| small_mels compute | 0.073–0.076 s | 0.069–0.073 s |
| default compute | 0.082–0.084 s | 0.076–0.082 s |
| large_fft compute | 0.146–0.153 s | 0.144–0.151 s |
| validate small_mels（三轮） | 0.184 / 0.182 / 0.184 | 0.183 / 0.178 / 0.179 |
| validate large_fft（三轮） | 0.355 / 0.357 / 0.360 | 0.349 / 0.355 / 0.355 |

Mel 阶段缩短约 30%，compute 缩短 2–7%；端到端 small_mels/large_fft 三轮都快 2–6 ms，default/narrow_band 在噪声内。结果不变、无副作用，采用。

### R5 实验：两帧合并为一次复数 FFT（job 230，单线程微基准）

利用 $z = x_1 + i x_2$ 一次 N 点复数 FFT 得到两帧的谱：$X_1[k] = (Z[k] + \overline{Z[N-k]})/2$，$X_2[k] = (Z[k] - \overline{Z[N-k]})/2i$。MKL 的 `DFTI_REAL_REAL`（分离实部、虚部）存储可以让实部、虚部两个输入指针直接指向 `samples` 中两帧的位置，同样无需拷贝；拆分时也只需算活跃 bin。微基准验证拆分与 r2c 的差异在 3e-13 以内，但速度：

| 每帧耗时（µs，3 次） | r2c（当前） | 两帧一次 c2c（REAL_REAL） |
|---|---:|---:|
| 8192 点 | 12.95–13.07 | 18.60–18.99 |
| 16384 点 | 37.7–39.1 | 40.2–40.7 |

（还没算拆分的开销）MKL 对分离存储复数 FFT 的优化明显不如 r2c，放弃。

### R5 优化：DFTI 输出改用 PERM 打包格式（微基准 job 233）

上面的微基准里，16384 点 r2c 每帧约 38 µs，是 8192 点 13 µs 的 2.9 倍，而 N log N 只应是约 2.15 倍。为了看看 MKL 的其他配置在这里是否更快，在集群单核上比较了几种 r2c 配置（重叠帧 hop=128，3 次取最好，µs/帧）：

| 配置 | 8192 点 | 16384 点 |
|---|---:|---:|
| out-of-place，CCE 复数存储（当前） | 15.02 | 37.2 |
| out-of-place，CCE，`DFTI_WORKSPACE=AVOID` | 12.22 | 36.3 |
| in-place，CCE（含拷贝） | 14.49 | 41.7 |
| in-place，PERM（含拷贝） | 13.88 | 40.5–40.7 |
| **out-of-place，PERM** | **10.99** | **29.8** |
| out-of-place，PERM，`DFTI_WORKSPACE=AVOID`（job 236 复测） | 11.00 | 29.7–30.1 |

PERM 是 MKL 的实数打包格式：输出 N 个实数，`Re X[0], Re X[N/2]`，然后依次是 `Re X[k], Im X[k]`（k=1..N/2−1）。第二次提交（job 236）的数值与第一次相差不到 1%；PERM 再加 `WORKSPACE=AVOID` 没有区别。它比 CCE 快 20–27%，原因应该是省掉了把实数结果展开成 N/2+1 个复数的那一步。

改动很小：k=1..N/2−1 在 PERM 与 CCE 中位置相同（`x[2k], x[2k+1]`），只有 X[0] 和 X[N/2] 的实部被打包到 `x[0], x[1]`（它们的虚部为 0）。所以变换后执行 `x[N] = x[1]; x[N+1] = 0; x[1] = 0;` 即可还原，频域加窗、共轭对称补边和 power 的代码都不用改。n_fft 已经被校验为大于 1 的 2 的幂，所以 N 一定是偶数，PERM 的偶数长度布局一定适用。只有频域加窗路径（double、非批量）使用 PERM，float 和批量路径维持原样。

prometheus 上：单元测试通过，误差余量与原来完全相同（0.0005/0.0008/0.0008/0.0006）；compute default 0.30 → 0.26–0.27 s（−12%），narrow_band 0.54 → 0.44 s（−18%）。集群对比见 job 235。

**集群 64 线程结果（job 235，hepnode2）——不采用。** 单核微基准的结论在满载下不成立：

| | 原版（CCE） | PERM |
|---|---|---|
| worker 0 FFT 阶段，default | 0.026–0.031 s | 0.032–0.035 s |
| compute small_mels / default | 0.071–0.075 / 0.077–0.081 | 0.078–0.082 / 0.086–0.087 |
| compute large_fft / narrow_band | 0.150–0.155 / 0.132–0.138 | 0.146–0.156 / 0.132–0.140 |
| validate default（三轮） | 0.284 / 0.284 / 0.287 | 0.291 / 0.289 / 0.296 |
| validate small_mels | 0.179 / 0.183 / 0.182 | 0.187 / 0.187 / 0.193 |
| validate large_fft | 0.368 / 0.365 / 0.359 | 0.352 / 0.357 / 0.362 |

8192 点的两组明显变慢，只有 large_fft 略快，narrow_band 没有差别（四组误差统计不变；small_mels 与 narrow_band 的输出甚至逐位相同）。单核时 PERM 快 27%，64 核满载时反而慢。一个可能的解释是两条代码路径在全核频率限制和共享缓存/内存带宽下表现不同，但没有验证。按 FFT 长度选择格式的证据不足（narrow_band 无差异），不采用，代码撤回。教训：**单线程微基准只能用来筛选候选，最终必须用满载的真实程序判断**。这也是本报告一直坚持用独占节点上的 `validate.py` 做最终判断的原因。

### R5 最终状态

| case | 原始实现 | R2 B_huge 64 核（job 207） | R4 最终（job 207） | **R5 最终（job 235，三轮）** | 相对 R4 |
|---|---:|---:|---:|---:|---:|
| default | 2058 s | 0.370 | 0.307 | **0.284–0.287** | −7% |
| small_mels | 562 s | 0.226 | 0.188 | **0.179–0.183** | −4% |
| large_fft | 4432 s | 0.450 | 0.403 | **0.359–0.368** | −10% |
| narrow_band | 未测 | 0.442 | 0.384 | **0.347–0.355** | −9% |

第五轮采用了三项：频域加窗、Mel 4 帧一组累加、两件串行准备工作分到两个 worker。其中前两项都在 compute 内部，第三项消除了 worker 0 的拖尾。default 0.285 s 的构成约为：读 0.07 s + compute 0.077 s + 写 0.14 s。compute 中 FFT 约占 3/4，MKL 的各种配置都已在满载下比较过，当前选择最快；读写都在 `main.cpp` 中，分配器能影响的部分已接近下限。

## 第六轮：按全 worker 的分阶段剖析继续压 compute

### 为什么不用 VTune / perf

集群上 `perf_event_paranoid=4`，普通用户不能用 `perf`；VTune 2026 的用户态采样（`sampling-mode=sw`）也启动失败：`Cannot start data collection because the scope of ptrace system call is limited`（`yama/ptrace_scope` 受限）。所以只能靠程序内插桩。

### 插桩扩展

原来的 `MELSPEC_TRACE` 只统计 worker 0 的阶段时间；从 R4 起 worker 0（以及 R5 起的 worker 1）还承担准备工作，已经不具代表性。改为每个 worker 都按 6 个阶段计时（window、fft、power、mel、store：写 tile、暂存和 flush、sched：领取任务和最后等待），汇总后输出平均每个 worker 的秒数。只有设置 `MELSPEC_TRACE=1` 时才计时，平时只多一次分支判断。

**集群结果（job 241，hepnode3，64 线程，每个 worker 平均，s）**

| case | kernel | fft | power | mel | store |
|---|---|---|---|---|---|
| small_mels | 0.068–0.074 | 0.048–0.053 | 0.0047 | 0.0064 | 0.001–0.005 |
| default | 0.080–0.083 | 0.050–0.052 | 0.0049 | 0.0068–0.0073 | 0.006–0.008 |
| large_fft | 0.147–0.160 | 0.102–0.112 | 0.0088 | 0.013 | 0.0045–0.0056 |
| narrow_band | 0.130–0.138 | 0.099–0.105 | 0.0034 | 0.0048 | 0.004–0.007 |

FFT 占 kernel 的 63–75%；power + Mel 约 15%；store 在 default 上约 9%（256 个 Mel 行，每个 tile 往相距 1.5 MB 的 256 处各写 128 字节，基本都是写缺失）；其余约 10% 是线程启动、plan 创建和尾部不均衡。

### 据此提出的两个候选

1. **DFTI 不使用额外工作区（`DFTI_WORKSPACE=DFTI_AVOID`，仍是 CCE 存储）。** 单核微基准中 8192 点从 15.0 降到 12.2 µs（−19%）。PERM 在满载下失效了，但这是另一项改动，需要单独在真实程序里检验。编译开关 `MELSPEC_DFTI_NO_WORKSPACE`。
2. **Mel 一组 8 帧。** FMA 延迟约 4 个周期、每周期可发射 2 条，要约 8 条独立依赖链才能填满流水线，4 帧一组还不够。累加改写为 `sums[kMelGroup]` 的通用形式（编译期展开），开关 `MELSPEC_MEL_GROUP`（默认 4）。

prometheus 上 4/8 帧一组以及不用工作区三种构建，四组参数的输出都与 HEAD **逐位相同**（不用工作区连舍入都没有改变），单元测试全部通过。集群对比见 job 242。

**结果（job 242，hepnode2，独占 64 核）——两项都不采用。** 四个构建在三组参数下的输出都与 HEAD 逐位相同。

| 每个 worker 平均（3 次，s） | 现状 g4 | 8 帧一组 g8 | 不用工作区 nw | g8 + nw |
|---|---|---|---|---|
| default fft / mel | 0.050–0.052 / 0.0070–0.0073 | 0.050–0.052 / 0.0073–0.0076 | 0.050 / 0.0069–0.0070 | 0.051 / 0.0073–0.0077 |
| default kernel | 0.078–0.081 | 0.078–0.084 | 0.076–0.079 | 0.078–0.085 |
| large_fft fft | 0.104–0.107 | 0.104–0.106 | 0.103–0.107 | 0.107–0.115 |

validate 三轮四组全部 PASS，四个变体互有胜负，都在 ±10 ms 噪声内（例如 default：g4 0.284/0.296/0.288，nw 0.299/0.289/0.284，g8nw 0.285/0.281/0.286）。

- 8 帧一组时 Mel 阶段没有变快：4 条链已经足够让 FMA 单元忙起来，Mel 不再受延迟限制。
- 不用工作区在单核微基准中快 19%，在 64 线程满载下 FFT 阶段没有区别，和 PERM 一样，单核收益没能延续到满载。

两个开关已删除。保留了全 worker 分阶段插桩（只在 `MELSPEC_TRACE=1` 时计时）和通用写法的 `sums[kMelGroup]` 累加（kMelGroup=4，输出与原来逐位相同）。

### R6 启动开销：worker 平均要 6–9 ms 才开始计算

在 trace 中加上所有 worker 的平均开始时间（`planned`）和平均完成时间后（job 243，hepnode2），发现平均每个 worker 要 **6–9 ms** 才开始计算（default 0.0065–0.0081 s，large_fft 0.0077–0.0091 s），而 default 整个 kernel 只有约 0.078 s，相当于浪费了 8–11% 的计算能力。负责准备工作的 worker 0/1 本来就晚开始（约 0.04 s），但它们只把平均值拉高约 1.2 ms，其余 62 个 worker 也平均要 5–7 ms。

为了分清原因，写了一个在全新进程中模拟启动过程的微基准（`start_bench.cpp`，job 245，64 线程，3 次）：

| 方式 | 8192：线程开始（均值/最大） | 8192：plan 就绪 | 16384：线程开始 | 16384：plan 就绪 |
|---|---|---|---|---|
| 每个 worker 各自 commit（现状） | 2.0–2.8 / 4.0–4.9 ms | 2.8–4.0 ms | 2.8–4.3 / 5.3–12.6 ms | 4.5–8.6 ms |
| 主线程 commit，worker `DftiCopyDescriptor` | 2.1–2.2 / 3.5–3.9 ms | 2.8–3.2 ms | 3.9–4.3 / 6.2–9.5 ms | 5.3–7.2 ms |
| 主线程 commit，所有 worker 共用一个描述符 | 1.9–2.1 / 3.3–3.8 ms | 2.1–2.2 ms | 2.2–2.6 / 3.5–5.2 ms | 2.7–3.3 ms |

两个来源：

1. **线程创建是串行的**：主线程逐个创建 63 个线程，平均要约 2 ms 才轮到，最后一个约 4 ms。
2. **每个 worker 首次 commit 描述符**：平均再加 1–4 ms，16384 点更明显，多个线程同时初始化时有明显争用。

对应的两项改动：

- **两级创建线程**：主线程先创建其他各组（每组约 √workers 个，64 线程时为 8）的组长，再创建本组成员；每个组长负责创建自己组的成员并在最后 join 它们。worker 的异常仍按槽位收集；创建线程失败时用互斥锁记录，等所有线程 join 之后再按原来的顺序抛出（和原来的区别是：创建失败时 worker 0 仍会执行自己的部分，因为任务是动态领取的，这样更安全）。
- **共享描述符**（开关 `MELSPEC_SHARED_PLAN`）：主线程 commit 一次，所有 worker 共用。DFTI 对已 commit 描述符的计算调用是线程安全的；仍然用逐字节比较来验证没有数据竞争。

prometheus 上 3 线程与 32 线程的输出都与 HEAD 逐字节相同。prometheus 这时整体变慢（HEAD 自身也从 0.27 s 变成 0.45 s），计时没有参考价值；只看启动，共享描述符把平均开始时间从 0.048–0.060 s 降到 0.028–0.032 s。集群对比见 job 247。

**集群结果（job 247，hepnode3，独占 64 核）——两项都不采用。** 三个构建在四组参数下的输出都与 HEAD 逐字节相同（共享描述符没有数据竞争）。

| 每个 worker 平均（3 次） | 现状 tr | 两级创建 tree | tree + 共享描述符 |
|---|---|---|---|
| default 平均开始时间 | 0.0072–0.0081 s | 0.0055–0.0061 s | 0.0056–0.0066 s |
| default kernel | 0.078–0.080 s | 0.079–0.081 s | 0.079–0.082 s |
| large_fft FFT 阶段 | 0.101–0.110 s | 0.105–0.117 s | **0.121–0.126 s** |
| validate default（三轮） | **0.276 / 0.281 / 0.280** | 0.286 / 0.292 / 0.284 | 0.291 / 0.285 / 0.281 |

- 两级创建确实让 worker 平均早开始约 2 ms，但 kernel 并没有变短，validate 三轮 default 反而慢 4–11 ms。原因：动态调度本来就会让早开始的 worker 多做一些，晚开始几毫秒的代价已经被摊平了；真正决定 kernel 时长的是总工作量和最后完成的时刻。
- 共享描述符让 FFT 阶段明显变慢（16384 点慢约 15%），推测是多个线程同时读写同一个描述符的内部状态，导致缓存行在核间来回迁移。微基准中只测了"就绪时间"，没有测之后的计算速度，所以没暴露这个问题。

两项都撤回，只保留平均开始/完成时间的 trace 输出（诊断用）。启动开销这一项的结论：表面上浪费 8–11%，实际上已被动态调度吸收，不是可回收的时间。

### R6 实验：tile 形状与每次领取的 tile 数（job 249，hepnode3）

store 在 default 上约占 9%：每个 tile 往 256 个 Mel 行各写 128 字节，行首不对齐（n_frames 是奇数），claim 边界上的缓存行会被两个 worker 写到。于是比较了 tile 帧数 × 每次领取 tile 数的四种组合：现状 16×2、16×4、32×1、32×2。输出全部与 HEAD 逐字节相同（12/12）。

| 3 次 | 16×2（现状） | 16×4 | 32×1 | 32×2 |
|---|---|---|---|---|
| default kernel | 0.075–0.078 | 0.075–0.079 | 0.076–0.079 | 0.073–0.079 |
| large_fft kernel | 0.141–0.155 | 0.144–0.147 | 0.146–0.149 | 0.138–0.144 |
| validate default（三轮） | 0.284 / 0.286 / 0.285 | 0.281 / 0.285 / 0.283 | 0.286 / 0.289 / 0.284 | 0.280 / 0.282 / 0.288 |
| validate large_fft | 0.350 / 0.358 / 0.370 | 0.349 / 0.358 / 0.348 | 0.353 / 0.353 / 0.358 | 0.347 / 0.361 / 0.347 |

全部在噪声内，store 阶段本身也没有稳定变化。store 的开销主要是对 394 MB 输出的写缺失（每个缓存行都要先读进来），与 tile 的形状关系不大。保留 16×2，不改。

### R6 小结

第六轮通过插桩找到了 compute 中剩下的每一块（FFT 63–75%、power+Mel 约 15%、store 约 5–9%、启动和尾部约 10%），并针对每一块做了满载实验：Mel 8 帧一组、DFTI 不用工作区、两级创建线程、共享描述符、四种 tile 形状。它们在真实程序中都与现状持平（共享描述符更慢）。这一轮没有改变默认构建（只增加了诊断用的 trace 输出），但用数据说明了：在 double 精度、不改 `main.cpp` 的前提下，compute 中已经没有可以回收的明显空间。剩下的大头是 MKL 的 FFT 本身，以及 `main.cpp` 的读写。

### R6 实验：`main.cpp` 的编译选项（job 252，hepnode2）

`main.cpp` 的源码不能改，但它的编译选项由 Makefile 决定。它的两个类型转换循环和 `vector` 清零都直接受编译选项影响。用 `io_breakdown`（逐句复制的 `main.cpp` 读写路径）比较（4 次，s）：

| 选项 | int16→double | out_f32 分配（含清零） | double→float |
|---|---:|---:|---:|
| 现状 `-O3 -march=native`（Ice Lake 默认偏好 256 位向量） | 0.0213–0.0216 | 0.0139–0.0142 | 0.0483–0.0494 |
| `-mprefer-vector-width=512` | 0.0237–0.0240 | 0.0140–0.0145 | 0.0485–0.0508 |
| `-fno-tree-loop-distribute-patterns` | 0.0212–0.0214 | 0.0243–0.0246 | 0.0484–0.0494 |

512 位向量让 int16→double 慢了 11%（降频大于宽度带来的收益）；禁止 GCC 把清零循环换成 `memset` 调用后，清零慢了 70%（glibc 的 `memset` 对大块内存使用非临时存储）。现状的选项已经是最好的，不改。顺带确认了集群上输入数据所在的 `/cache` 是 BeeGFS（`fhgfs`），读 49 MB 约 0.028 s，这是文件系统的速度。

### 考虑过但没有做的：替换 `memset`

`main.cpp` 的串行路径上还有约 35 ms 的清零：`vector` 值初始化对 `buf`（49 MB）、`y`（98 MB）、`out_f32`（197 MB）各做一次 `memset(0)`，而这些内存刚从分配器拿到，本来就是全零的。理论上可以在 `large_alloc.cpp` 中定义一个同名的 `memset` 覆盖 libc 的版本：遇到"刚分配、还没写过、要清成 0"的块时直接返回，最多能省掉约 13% 的端到端时间。

没有做的原因：分配器无法知道从分配到 `memset` 之间这块内存有没有被写过（例如 `new` 之后先写几个字节再整体清零），要保证正确就得用 `mprotect` 把块设成只读，再用 SIGSEGV 处理程序拦截第一次写入。这等于在全局替换 C 库的基本函数，并依赖信号处理来维持正确性，风险和可维护性都不可接受，也超出了"优化 Mel 谱计算"的合理范围。记录在这里作为一个有意放弃的方向。

## 第七轮：让没有 MKL 时的回退路径也受益

### 动机

Makefile 只在默认的 spack 路径下找到 `mkl_dfti.h` 时才启用 MKL。如果评测环境没有这个路径（或 MKL 版本、路径不同），构建会**静默退回**第一到三轮的手写 FFT 路径。这条路径在 R4、R5 中没有改动，仍然用静态连续分块分配帧，R4 的 profile 已经说明静态分块在 64 线程下最慢和最快的 worker 相差 50% 以上。为了让性能不依赖环境，把 R4 中已验证的调度和输出写法移植到回退路径。

**基线（job 253，`MKL=0`，独占 64 核）**：validate 0.379 / 0.222 / 0.462 / 0.455 s（default/small_mels/large_fft/narrow_band），compute 波动很大（narrow_band 两次分别为 0.29 s 和 0.22 s），`output.resize` 的 0.025 s 仍在串行路径上。

### R7 步骤 1：动态领取 lane group（job 254）

回退路径按 SIMD lane group（每组 `kFrameLanes` 帧，跨帧向量化）处理。改为各 worker 每次从原子计数器领取 8 组；每个 worker 的 FFT 草稿缓冲区（约 400 KB）提到循环外只分配一次。如果每次领取都重新分配，会超过 glibc 的 mmap 阈值，导致每次领取都 mmap 并缺页。四组参数的输出与改动前逐字节相同。

| compute（3 次，s） | 静态分块 | 动态领取 |
|---|---|---|
| default | 0.157–0.159 | **0.122–0.123** |
| large_fft | 0.234–0.310 | **0.224–0.230** |
| narrow_band | 0.216–0.288 | **0.216–0.217** |

default 的 compute 缩短 22%，16384 点两组的大幅波动消失。validate（取 5 次的最小值，会掩盖一部分静态分块的波动）三轮中 12 项比较有 9 项更快，例如 default 0.347/0.348/0.367 → 0.353/0.343/0.354，narrow_band 0.444/0.458/0.446 → 0.436/0.445/0.439。

### R7 步骤 2：tile 化输出 + 输出清零与计算重叠（job 255）

回退路径原来每算完一个 lane group（4 帧）就往 256 个 Mel 行各写 32 字节，比 MKL 路径的 16 帧 tile（每行 128 字节）还要分散；而且 `output.resize()` 的 0.025 s 单线程清零在计算开始前串行执行。改为与 MKL 路径相同的做法：每次领取的 32 帧结果先写入 worker 私有的 `[mel][32]` tile，再逐行连续写出；worker 0 负责 resize（同样用 release/acquire 发布），其他 worker 在输出就绪之前把 tile 暂存起来。n_fft=1（窗为全零）的特例仍然直接 resize 并清零。prometheus 上 1/3/32 线程、四组参数共 12 种组合的输出都与 HEAD 的回退路径逐字节相同，`SCALAR=1` 的单元测试也通过；default 0.52–0.54 → 0.48–0.50 s。

**集群结果（job 255，hepnode2，独占 64 核，`MKL=0`）。** 四组输出与原回退路径逐字节相同。

| compute（3 次，s） | 静态分块 | 动态领取 | 动态 + tile + 重叠 |
|---|---|---|---|
| default | 0.123–0.157 | 0.122–0.124 | **0.106–0.109** |
| large_fft | 0.229–0.307 | 0.226–0.230 | **0.208–0.212** |
| narrow_band | 0.216–0.327 | 0.213–0.219 | **0.195–0.201** |

| validate 三轮（s） | 静态分块 | 动态领取 | 动态 + tile + 重叠 |
|---|---|---|---|
| default | 0.359 / 0.346 / 0.351 | 0.348 / 0.348 / 0.345 | **0.337 / 0.318 / 0.334** |
| small_mels | 0.227 / 0.226 / 0.229 | 0.222 / 0.221 / 0.224 | **0.219 / 0.215 / 0.215** |
| large_fft | 0.450 / 0.450 / 0.461 | 0.452 / 0.457 / 0.458 | **0.426 / 0.439 / 0.425** |
| narrow_band | 0.491 / 0.454 / 0.459 | 0.438 / 0.447 / 0.442 | **0.412 / 0.422 / 0.426** |

没有 MKL 时，端到端快 5–10%，compute 快 10–30% 且不再波动。两步都采用。回退路径与 MKL 路径的差距从 default 0.35 vs 0.28 s 缩小到 0.33 vs 0.28 s，剩下的差别主要是 FFT 本身（手写跨帧 radix-2 与 MKL）以及 MKL 路径特有的频域加窗。

### R7 步骤 3：第三轮遗留的 FFT 开关在新回退路径上的正式评估（job 256，hepnode2）

第三轮实现了两个手写 FFT 的开关，但当时只完成了构建和单元测试，没有正式评估：`FUSED_FFT`（把相邻两级 radix-2 合并成一趟，少一半对 FFT 数组的读写）和 `AVX512`（每组 8 帧，用 512 位向量）。回退路径有了动态调度之后，在独占 64 核上补做评估（`MKL=0`）：

| validate 三轮（s） | 现状 | **FUSED_FFT=1** | AVX512=1 | 两者都开 |
|---|---|---|---|---|
| default | 0.320 / 0.330 / 0.323 | **0.314 / 0.305 / 0.312** | 0.367 / 0.360 / 0.368 | 0.329 / 0.339 / 0.344 |
| small_mels | 0.215 / 0.219 / 0.218 | **0.191 / 0.190 / 0.193** | 0.214 / 0.209 / 0.215 | 0.189 / 0.188 / 0.185 |
| large_fft | 0.424 / 0.431 / 0.444 | **0.381 / 0.396 / 0.379** | 0.485 / 0.482 / 0.489 | 0.421 / 0.418 / 0.423 |
| narrow_band | 0.417 / 0.430 / 0.426 | **0.369 / 0.380 / 0.368** | 0.461 / 0.473 / 0.482 | 0.410 / 0.414 / 0.416 |

compute：FUSED_FFT 让 default 从 0.107 降到 0.085–0.088 s，large_fft 从 0.21 降到 0.162–0.166 s，narrow_band 从 0.197–0.199 降到 0.150–0.155 s（−17~23%）。它使用原来的 twiddle 表和完全相同的复数乘法表达式，prometheus 上四组参数的输出与不融合版本**逐字节相同**，误差余量也完全一致。AVX512 的 8 帧一组反而慢 20–30%：64 核同时执行 512 位指令时频率下降，抵消了宽度带来的收益，这和 R5 中 MKL 强制 AVX2 的观察可以对照着看（MKL 自己会权衡，手写代码则直接吃到降频）。

因此把 Makefile 中 `FUSED_FFT` 的默认值改为 1（只影响没有 MKL 时的手写路径），`AVX512` 保持 0。至此，没有 MKL 时的 default 从本轮开始时的 0.35 s 降到约 0.31 s，与 MKL 路径（约 0.28 s）的差距缩小到 10% 左右。
