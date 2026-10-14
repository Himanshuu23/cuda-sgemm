# Thin convenience wrapper around CMake. (On Windows without make, run the cmake
# commands in the `build` target by hand.)
BUILD ?= build

.PHONY: all build run profile plot emu-test clean

all: build

build:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD) -j

run: build
	./$(BUILD)/sgemm_bench

profile: build
	bash scripts/profile.sh

plot:
	python3 scripts/plot.py

# CPU-only check of every kernel's index math (needs g++ with C++20; no GPU).
emu-test:
	mkdir -p $(BUILD)
	g++ -std=c++20 -O2 -fno-strict-aliasing -pthread -Wno-unknown-pragmas -Wno-dangling-pointer \
	    tests/cpu_emulation_test.cpp -o $(BUILD)/emu_test
	./$(BUILD)/emu_test

clean:
	rm -rf $(BUILD)
