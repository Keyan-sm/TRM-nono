import numpy as np
import os
import argparse
import sys

def calculate_clues(grid):
    """
    Calculates row and column clues for a binary nonogram grid.
    
    Args:
        grid (numpy.ndarray): A 2D binary array (0s and 1s).
        
    Returns:
        tuple: (row_clues, col_clues) where each is a list of lists of integers.
    """
    rows, cols = grid.shape
    row_clues = []
    col_clues = []
    
    # Calculate row clues
    for r in range(rows):
        row = grid[r]
        clues = []
        count = 0
        for cell in row:
            if cell == 1:
                count += 1
            elif count > 0:
                clues.append(count)
                count = 0
        if count > 0:
            clues.append(count)
        row_clues.append(clues if clues else [0]) # Use [0] for empty rows if that's the convention, or []
        
    # Calculate column clues
    for c in range(cols):
        col = grid[:, c]
        clues = []
        count = 0
        for cell in col:
            if cell == 1:
                count += 1
            elif count > 0:
                clues.append(count)
                count = 0
        if count > 0:
            clues.append(count)
        col_clues.append(clues if clues else [0])
        
    return row_clues, col_clues

def get_nonograms(n, m, base_path='TRM Context/NonoDataset-main', random_selection=False):
    """
    Returns n nonograms of m dimension with their clues.
    
    Args:
        n (int): Number of nonograms to return.
        m (int): Dimension of the nonogram (m x m).
        base_path (str): Path to the NonoDataset-main directory.
        random_selection (bool): If True, randomly select n nonograms.
        
    Returns:
        list: A list of dictionaries, each containing:
              - 'grid': numpy.ndarray (m x m)
              - 'row_clues': list of lists
              - 'col_clues': list of lists
    """
    
    # Map dimension to file path
    # Note: Paths are relative to the project root or provided base_path
    if m == 5:
        file_path = os.path.join(base_path, '5x5', 'target_combined.npz')
    elif m == 10:
        file_path = os.path.join(base_path, '10x10', 'y_train_dataset.npz')
    elif m == 15:
        # Using test dataset as it is readily available as npz. 
        # Train dataset is zipped.
        file_path = os.path.join(base_path, '15x15', 'y_test_15x15_ok.npz')
    else:
        raise ValueError(f"Unsupported dimension: {m}. Only 5, 10, and 15 are supported.")
        
    if not os.path.exists(file_path):
        # Try absolute path if relative fails
        abs_base_path = os.path.abspath(base_path)
        file_path = os.path.join(abs_base_path, '5x5' if m==5 else '10x10' if m==10 else '15x15', 
                                 'target_combined.npz' if m==5 else 'y_train_dataset.npz' if m==10 else 'y_test_15x15_ok.npz')
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        with np.load(file_path) as data:
            # Assuming the key is 'arr_0' based on inspection
            if 'arr_0' in data:
                dataset = data['arr_0']
            else:
                # Fallback to the first key
                dataset = data[data.files[0]]
                
            total_available = dataset.shape[0]
            
            if n > total_available:
                print(f"Warning: Requested {n} nonograms, but only {total_available} are available. Returning all.")
                n = total_available
            
            # Select n nonograms
            if random_selection:
                # Randomly select n indices without replacement
                indices = np.random.choice(total_available, n, replace=False)
                selected = dataset[indices]
            else:
                # Take the first n
                selected = dataset[:n]
            
            # Reshape to (n, m, m)
            reshaped = selected.reshape(n, m, m)
            
            results = []
            for i in range(n):
                grid = reshaped[i]
                row_clues, col_clues = calculate_clues(grid)
                results.append({
                    'grid': grid,
                    'row_clues': row_clues,
                    'col_clues': col_clues
                })
            
            return results

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def print_nonogram(data):
    """Prints a nonogram grid and clues to stdout."""
    grid = data['grid']
    row_clues = data['row_clues']
    col_clues = data['col_clues']
    
    print("Row Clues:", row_clues)
    print("Col Clues:", col_clues)
    print("Grid:")
    for row in grid:
        print(" ".join(["■" if cell else "□" for cell in row]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get n nonograms of m dimension.")
    parser.add_argument("n", type=int, help="Number of nonograms")
    parser.add_argument("m", type=int, help="Dimension (5, 10, or 15)")
    parser.add_argument("--path", type=str, default="TRM Context/NonoDataset-main", help="Path to dataset")
    parser.add_argument("-r", "--random", action="store_true", help="Randomly select nonograms")
    
    args = parser.parse_args()
    
    try:
        nonograms = get_nonograms(args.n, args.m, args.path, args.random)
        
        if nonograms is not None:
            print(f"Successfully loaded {len(nonograms)} nonograms of size {args.m}x{args.m}.")
            print("-" * 20)
            print("First nonogram sample:")
            print_nonogram(nonograms[0])
            print("-" * 20)
            
    except Exception as e:
        print(f"Error: {e}")
