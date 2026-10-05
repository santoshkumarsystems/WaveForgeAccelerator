# WaveForge Accelerator Roadmap

WaveForge Accelerator already has a validated end-to-end AI model → compiler → runtime → accelerator-style execution path.

This roadmap separates **completed capabilities** from future compiler/runtime depth so the project does not understate completed work or claim unfinished features.

## Completed foundation

### AI model and data pipeline

- [x] cuWaves physical simulation as the data source
- [x] deterministic 10,000-example dataset generation
- [x] **AI model training — neural network trained from scratch in PyTorch**
- [x] amplitude + wavelength regression
- [x] deterministic training/validation workflow
- [x] trained FP32 model weights

### Portable model

- [x] ONNX export
- [x] external ONNX initializer-data handling
- [x] PyTorch ↔ ONNX Runtime numerical parity
- [x] ONNX graph / initializer inspection

### Real inference-runtime references

- [x] ONNX Runtime baseline
- [x] OpenVINO real CPU inference
- [x] TensorRT real NVIDIA Tesla T4 inference
- [x] TensorRT ↔ ONNX Runtime parity
- [x] measured OpenVINO / TensorRT reference timings

### C++ accelerator runtime

- [x] task scheduling / execution plan
- [x] memory management
- [x] Host → Device Memory → SRAM hierarchy
- [x] cold/warm model residency
- [x] logical DMA-style Device Memory → SRAM movement
- [x] kernel dispatch
- [x] real FP32 GEMM / ReLU / Mul / Add execution
- [x] profiler
- [x] deterministic logical device-memory map
- [x] device / SRAM address-contract validation

### SRAM-aware execution

- [x] transient-memory reservation
- [x] initializer staging-capacity calculation
- [x] 16 KiB whole-model residency path
- [x] 4 KiB constrained-SRAM path
- [x] real row-aligned GEMM tiling
- [x] validated `13 + 13 + 6` first-GEMM tile geometry

### Apache TVM compiler integration

- [x] ONNX → Relax
- [x] Relax → PrimFunc / TIR inspection
- [x] compiler-side WaveForge target contract
- [x] SRAM-aware scheduling policy
- [x] BYOC pattern matching / partitioning
- [x] five `Codegen="waveaccel"` partitions
- [x] full current-model BYOC graph coverage
- [x] BYOC → ExecutionPlan bridge
- [x] integrated BYOC + SRAM-aware compiler-plan emission
- [x] Python compiler target contract transported into C++ Runtime
- [x] compiler/manual schedule parity

### Validation and reproducibility

- [x] 13/13 C++ regression tests
- [x] layer-by-layer validation matrix
- [x] public CMake FetchContent path
- [x] pinned cuWaves v0.0.1 dependency
- [x] public dependency build validated without local cuWaves override
- [x] TensorRT environment and reproducibility instructions
- [x] real-vs-simulated claims separation

## Next compiler milestone

### Native TVM external-codegen path

- [ ] implement a real TVM `RunCodegen` backend
- [ ] register WaveForge external code generation
- [ ] replace the current BYOC → ExecutionPlan bridge with a more native TVM codegen invocation where appropriate
- [ ] return a backend artifact/runtime representation through TVM
- [ ] add `RunCodegen` integration tests
- [ ] preserve current SRAM-aware ExecutionPlan and DeviceCommand semantics

This is the most natural next compiler-depth milestone because BYOC partitioning, target contracts, SRAM planning, and the C++ runtime already exist.

## Future accelerator-runtime depth

- [ ] concurrent command / request scheduling
- [ ] explicit dependency/event model
- [ ] asynchronous command queues
- [ ] pipelined Device Memory ↔ SRAM movement and compute
- [ ] richer scheduling policies
- [ ] multi-buffer / double-buffer SRAM experiments
- [ ] additional operator support
- [ ] richer graph fusion experiments

## Future compiler depth

- [ ] native TVM `TargetKind` exploration
- [ ] custom TVM runtime Module exploration
- [ ] deeper TensorIR scheduling
- [ ] tensorization experiments where meaningful
- [ ] quantization
- [ ] compiler-driven operator fusion
- [ ] optional LLVM / MLIR backend experiments

These are future capabilities and are not claimed by the current implementation.

## Future hardware-facing depth

- [ ] real accelerator driver interface
- [ ] real DMA / PCIe integration
- [ ] real device queues / interrupts / synchronization
- [ ] real device memory allocation
- [ ] hardware-specific kernel binaries / ISA
- [ ] measured accelerator bandwidth / latency

The current WaveForge backend intentionally uses logical addresses and simulated accelerator timing.

## Optional research directions

- [ ] analog / compute-in-memory non-ideality modeling
- [ ] weight-stationary/dataflow mapping experiments
- [ ] fixed compute-array dimensions
- [ ] energy/cost modeling
- [ ] larger trained models and more complex graphs

These are research directions, not current project claims.
