import os
import shutil
import sys

def import_dataset(downloaded_folder_path):
    """
    This script will take a downloaded Kaggle dataset folder and safely copy 
    all the Real and Fake images into your project's Data folder.
    """
    project_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    project_real_dir = os.path.join(project_data_dir, "REAL")
    project_fake_dir = os.path.join(project_data_dir, "FAKE")

    os.makedirs(project_real_dir, exist_ok=True)
    os.makedirs(project_fake_dir, exist_ok=True)

    print(f"Scanning downloaded folder: {downloaded_folder_path}")
    
    real_count = 0
    fake_count = 0

    for root, dirs, files in os.walk(downloaded_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                
                # Try to guess if it's real or fake based on the folder name it came from
                folder_name = os.path.basename(root).lower()
                
                if 'real' in folder_name or 'original' in folder_name or 'human' in folder_name:
                    dest_path = os.path.join(project_real_dir, f"imported_real_{real_count}.jpg")
                    shutil.copy2(source_path, dest_path)
                    real_count += 1
                elif 'fake' in folder_name or 'manipulated' in folder_name or 'forged' in folder_name or 'ai' in folder_name or 'synthetic' in folder_name:
                    dest_path = os.path.join(project_fake_dir, f"imported_fake_{fake_count}.jpg")
                    shutil.copy2(source_path, dest_path)
                    fake_count += 1

    print("=========================================")
    print("IMPORT COMPLETE!")
    print(f"Successfully added {real_count} NEW REAL faces.")
    print(f"Successfully added {fake_count} NEW FAKE faces.")
    print("=========================================")
    print("You are now ready to run: python backend/train.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_data.py <path_to_downloaded_dataset_folder>")
        print("Example: python import_data.py C:/Users/Downloads/archive")
    else:
        import_dataset(sys.argv[1])
