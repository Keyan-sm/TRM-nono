# Tiny Recursive Model vs Frontier Models on Nonogram Puzzles
**Authors:** Keyan Mikaili, Samik Bhinge, Andre Keyser
## Abstract
This project evaluates how a Tiny Recursive Model (TRM) with only 7M parameters compares to Gemini 3 Pro Thinking on nonogram logic puzzles. Using a dataset of 360K 10x10 and 660K 15x15 nonograms, we trained our TRM and benchmarked it against the SOTA model.
**Key Results:**
- **10x10**: TRM achieved 96% exact match vs Gemini's 87%
- **15x15**: TRM achieved 82% exact match vs Gemini's 52% with 70x faster inference
Our findings demonstrate that narrowly trained recursive reasoning models can outperform generalized SOTA models on specific deterministic logic tasks while being vastly more efficient.
## Repository Structure
```
├── TRM_PoC_10x10/       # 10x10 nonogram TRM implementation
├── TRM_PoC_15x15/       # 15x15 nonogram TRM implementation (with stability fixes)
├── TRM Context/         # Samsung TRM base code and context materials
├── Deliverable/         # Final report and rubric
└── *.py                 # Utility and testing scripts
```
| Document | Location |
|----------|----------|
| **10x10 TRM Code** | `TRM_PoC_10x10/` |
| **15x15 TRM Code** | `TRM_PoC_15x15/` |
| **Gemini Test Script** | `test_gemini_nonograms.py` |
