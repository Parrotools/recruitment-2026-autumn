CXX ?= c++
CPPFLAGS ?=
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O3
LDFLAGS ?=
LDLIBS ?=

SCALAR ?= 0
HUGE_ALLOC ?= 0
FUSED_FFT ?= 0
AVX512 ?= 0
ARCH_FLAGS ?=
CLANG_FORMAT ?= clang-format
MELSPEC_TEST_TMPDIR ?= $(abspath $(CURDIR)/../melspec-optimization-artifacts)
ROUND2_BUILD_DIR ?= $(abspath $(CURDIR)/../melspec-optimization-20261029/round2/build)

ifneq ($(filter-out 0 1,$(SCALAR) $(HUGE_ALLOC) $(FUSED_FFT) $(AVX512)),)
$(error SCALAR, HUGE_ALLOC, FUSED_FFT, and AVX512 must be 0 or 1)
endif

MELSPEC_DEFINES =
SCALAR_CPPFLAG = $(filter -DMELSPEC_SCALAR=%,$(CPPFLAGS))
HUGE_CPPFLAG = $(filter -DMELSPEC_HUGE_ALLOC=%,$(CPPFLAGS))
FUSED_CPPFLAG = $(filter -DMELSPEC_FUSED_FFT=%,$(CPPFLAGS))
AVX512_CPPFLAG = $(filter -DMELSPEC_AVX512=%,$(CPPFLAGS))
ifneq ($(SCALAR_CPPFLAG),)
ifneq ($(SCALAR_CPPFLAG),-DMELSPEC_SCALAR=$(SCALAR))
$(error CPPFLAGS conflicts with SCALAR=$(SCALAR); use the SCALAR make option)
endif
else
MELSPEC_DEFINES += -DMELSPEC_SCALAR=$(SCALAR)
endif
ifneq ($(HUGE_CPPFLAG),)
ifneq ($(HUGE_CPPFLAG),-DMELSPEC_HUGE_ALLOC=$(HUGE_ALLOC))
$(error CPPFLAGS conflicts with HUGE_ALLOC=$(HUGE_ALLOC); use the HUGE_ALLOC make option)
endif
else
MELSPEC_DEFINES += -DMELSPEC_HUGE_ALLOC=$(HUGE_ALLOC)
endif
ifneq ($(FUSED_CPPFLAG),)
ifneq ($(FUSED_CPPFLAG),-DMELSPEC_FUSED_FFT=$(FUSED_FFT))
$(error CPPFLAGS conflicts with FUSED_FFT=$(FUSED_FFT); use the FUSED_FFT make option)
endif
else
MELSPEC_DEFINES += -DMELSPEC_FUSED_FFT=$(FUSED_FFT)
endif
ifneq ($(AVX512_CPPFLAG),)
ifneq ($(AVX512_CPPFLAG),-DMELSPEC_AVX512=$(AVX512))
$(error CPPFLAGS conflicts with AVX512=$(AVX512); use the AVX512 make option)
endif
else
MELSPEC_DEFINES += -DMELSPEC_AVX512=$(AVX512)
endif

MELSPEC_COMPILE_FLAGS = $(CXXFLAGS) $(ARCH_FLAGS) -pthread
LDLIBS += -pthread

ifeq ($(AVX512),1)
ifeq ($(SCALAR),0)
ifeq ($(findstring -mavx512f,$(CXXFLAGS) $(ARCH_FLAGS)),)
$(error AVX512=1 requires -mavx512f in CXXFLAGS or ARCH_FLAGS)
endif
endif
endif

ifeq ($(HUGE_ALLOC),1)
ifneq ($(shell uname -s),Linux)
$(error HUGE_ALLOC=1 is supported only on Linux)
endif
endif

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
OBJS = $(SRCS:.cpp=.o)
EXEC = melspec

.PHONY: all clean test unit-test allocator-test format benchmark-compute

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(MELSPEC_COMPILE_FLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.cpp $(INCS)
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

unit-test:
	@tmp_dir="$$TMPDIR"; \
	if [ -z "$$tmp_dir" ]; then tmp_dir="$(MELSPEC_TEST_TMPDIR)"; fi; \
	mkdir -p "$$tmp_dir"; \
	test_binary=$$(mktemp "$$tmp_dir/melspec-unit.XXXXXX"); \
	trap 'rm -f "$$test_binary"' EXIT; \
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include tests/test_melspec.cpp src/melspec.cpp \
		$(LDFLAGS) $(LDLIBS) -o "$$test_binary" && "$$test_binary"

ifeq ($(HUGE_ALLOC),1)
allocator-test:
	@tmp_dir="$$TMPDIR"; \
	if [ -z "$$tmp_dir" ]; then tmp_dir="$(MELSPEC_TEST_TMPDIR)"; fi; \
	mkdir -p "$$tmp_dir"; \
	test_binary=$$(mktemp "$$tmp_dir/melspec-allocator.XXXXXX"); \
	trap 'rm -f "$$test_binary"' EXIT; \
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include tests/test_large_alloc.cpp \
		src/large_alloc.cpp $(LDFLAGS) $(LDLIBS) -o "$$test_binary" && "$$test_binary"
else
allocator-test:
	@echo "Run with HUGE_ALLOC=1 on Linux: make clean && make HUGE_ALLOC=1 allocator-test"
	@exit 1
endif

benchmark-compute:
	@mkdir -p "$(ROUND2_BUILD_DIR)"
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include -Dmain=melspec_cli_main \
		-c src/main.cpp -o "$(ROUND2_BUILD_DIR)/main_io.o"
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include \
		-c src/melspec.cpp -o "$(ROUND2_BUILD_DIR)/melspec.o"
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include \
		-c src/large_alloc.cpp -o "$(ROUND2_BUILD_DIR)/large_alloc.o"
	$(CXX) $(CPPFLAGS) $(MELSPEC_DEFINES) $(MELSPEC_COMPILE_FLAGS) -I./include \
		tests/benchmark_compute.cpp "$(ROUND2_BUILD_DIR)/main_io.o" \
		"$(ROUND2_BUILD_DIR)/melspec.o" "$(ROUND2_BUILD_DIR)/large_alloc.o" \
		$(LDFLAGS) $(LDLIBS) \
		-o "$(ROUND2_BUILD_DIR)/benchmark_compute"

format:
	$(CLANG_FORMAT) -i src/melspec.cpp src/large_alloc.cpp tests/test_melspec.cpp tests/test_large_alloc.cpp tests/benchmark_compute.cpp
