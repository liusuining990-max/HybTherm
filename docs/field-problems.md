# Handling problems behind this project

The companion repository lists 20 kinds of process and handling problem in NGS testing workflows: [OpenNGS-Bench/docs/field-problems.md](https://github.com/liusuining990-max/OpenNGS-Bench/blob/main/docs/field-problems.md).

The rows below concern the 65 °C hybridization-capture step, which is what this repository models and measures.

| Event | Problem | Fix | Result | Principle |
|---|---|---|---|---|
| Capture performance | Capture efficiency varies from run to run; eluted library concentration is too low | Control library quality, hybridization time and **temperature**, blockers, input, wash stringency and amplification cycles | More consistent on-target rate | 65 °C balances specificity and yield: lower lets mismatched fragments stay bound, higher releases true targets |
| Incubator performance | Instruments advertise temperature uniformity, but independent measurements are rarely published | Measure the block with calibrated probes | Known, not assumed, well-to-well differences | A single control sensor reports one point; wells at the edges run cooler |

**The gap:** block temperature is named as a factor and advertised as a specification, but is rarely measured independently. This repository provides the model, the analysis pipeline and the logging tools to make that measurement.
