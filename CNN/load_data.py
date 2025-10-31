import numpy as np
import os

def load_data():
    """
    Load and extract data from images_malware.npz

    Returns:
        dict: Dictionary containing the arrays stored in the .npz file
    """
    # Load the .npz file with allow_pickle=True to handle object arrays
    data = np.load("C:\\Users\\Utilisateur\\OneDrive - IPSA\\A5 S1\\Ma513\\FINAL_PROJECT\\FINAL_PROJECT\\images_malware.npz", allow_pickle=True)

    # Print the keys (array names) in the file
    print("Arrays in the file:", data.files)

    # Create a dictionary to store the extracted arrays
    extracted_data = {}

    # Extract each array and store it in the dictionary
    for key in data.files:
        extracted_data[key] = data[key]
        print(f"Extracted array '{key}' with shape {extracted_data[key].shape}")

    return extracted_data

def save_extracted_data(extracted_data, output_dir="extracted_data"):
    """
    Save the extracted data to separate files

    Args:
        extracted_data (dict): Dictionary containing the extracted arrays
        output_dir (str): Directory to save the extracted data
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save each array to a separate file
    for key, array in extracted_data.items():
        output_file = os.path.join(output_dir, f"{key}.npy")
        np.save(output_file, array)
        print(f"Saved array '{key}' to {output_file}")

if __name__ == "__main__":
    # Extract the data when the script is run directly
    extracted_data = load_data()

    # Additional information about the extracted data
    for key, array in extracted_data.items():
        print(f"\nDetailed information about '{key}':")
        print(f"  - Data type: {array.dtype}")
        print(f"  - Number of elements: {array.size}")
        print(f"  - Shape: {array.shape}")
        if len(array) > 0:
            print(f"  - First element preview: {array[0]}")

    print("\nExtraction complete!")