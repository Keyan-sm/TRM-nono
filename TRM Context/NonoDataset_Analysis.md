# Nonogram Dataset Analysis

This document provides a detailed explanation of the `NonoDataset-main` dataset found in the `TRM Context` directory. The dataset contains Nonogram puzzles (also known as Picross or Griddlers) of various dimensions, split into training and testing sets.

## 1. Directory Structure

The dataset is organized by grid dimension:

- **`5x5/`**: Contains data for 5x5 puzzles.
- **`10x10/`**: Contains data for 10x10 puzzles.
- **`15x15/`**: Contains data for 15x15 puzzles.

## 2. File Naming Convention

The files generally follow this naming convention:

- **Prefix**:
    - `x_`: Represents the **Input** data (the clues/constraints).
    - `y_`: Represents the **Target** data (the solved binary grid).
    - `database_`: Indicates a specific subset of data (e.g., icons, noise).

- **Suffix/Type**:
    - `_train_`: Training dataset.
    - `_test_`: Testing/Validation dataset.
    - `_icons`: Puzzles that form recognizable images (likely human-designed).
    - `_noise`: Puzzles that are likely procedurally generated or random patterns.
    - `_combined`: Aggregated data.

- **Extension**:
    - `.npz`: Numpy archive files containing compressed arrays.
    - `.zip`: Zipped archives (likely containing `.npz` or raw data).

## 3. Data Format

The data is stored as Numpy arrays within `.npz` files. The primary key in these archives is usually `arr_0`.

### Inputs (Clues) - `x_*.npz`

The input data represents the row and column clues for the Nonogram puzzles.
- **Format**: Flattened integer array.
- **Structure**: `[Row 1 Clues, Row 2 Clues, ..., Col 1 Clues, Col 2 Clues, ...]`
- **Padding**: Each row/column constraint is padded with zeros to a fixed length (slots) to ensure a uniform array shape.

| Dimension | Slots per Constraint | Total Input Length | Calculation |
| :--- | :--- | :--- | :--- |
| **10x10** | 5 | **100** | (10 rows + 10 cols) * 5 slots |
| **15x15** | 8 | **240** | (15 rows + 15 cols) * 8 slots |

**Example (10x10)**:
A sample input `[0, 0, 0, 0, 10, ...]` implies the first row has a single clue of `10`.
A sample input `[0, 0, 3, 1, 2, ...]` implies a row with clues `3`, `1`, `2`.

### Targets (Grids) - `y_*.npz`

The target data represents the solved Nonogram grid.
- **Format**: Flattened binary array (0s and 1s).
- **Structure**: Row-major flattened grid.
- **Values**: `1` represents a filled cell, `0` represents an empty cell.

| Dimension | Grid Size | Total Target Length |
| :--- | :--- | :--- |
| **5x5** | 5x5 | **25** |
| **10x10** | 10x10 | **100** |
| **15x15** | 15x15 | **225** |

## 4. Dataset Contents

### 10x10 Directory
- **`x_train_dataset.npz` / `y_train_dataset.npz`**: Main training set (~360k samples).
- **`database_icons.npz` / `database_y_icons.npz`**: Icon-based puzzles (~42k samples).
- **`database_noise.npz` / `database_y_noise.npz`**: Noise-based puzzles.

### 15x15 Directory
- **`x_test_15x15_ok.npz` / `y_test_15x15_ok.npz`**: Test set (~15k samples).
- **`x_train_15x15_ok.zip`**: Zipped training data.

### 5x5 Directory
- **`target_combined.npz`**: Contains ~33.5 million 5x5 grids.
