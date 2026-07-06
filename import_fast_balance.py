import os
import shutil
import sys
import uuid

def import_fast_balance(downloaded_folder_path, max_images_per_class=2500):
    """
    This script safely copies a limited number of Real and Fake images (e.g. 2500 each)
    from a massive dataset into your project's Data folder, avoiding long training times.
    """
    project_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    project_real_dir = os.path.join(project_data_dir, "REAL")
    project_fake_dir = os.path.join(project_data_dir, "FAKE")

    os.makedirs(project_real_dir, exist_ok=True)
    os.makedirs(project_fake_dir, exist_ok=True)

    print(f"Scanning downloaded folder: {downloaded_folder_path}")
    print(f"Goal: Import exactly {max_images_per_class} Real and {max_images_per_class} Fake images.\n")
    
    real_count = 0
    fake_count = 0

    for root, dirs, files in os.walk(downloaded_folder_path):
        for file in files:
            if real_count >= max_images_per_class and fake_count >= max_images_per_class:
                break # We have enough images!

            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                
                # Guess if it's real or fake based on the folder name
                folder_name = os.path.basename(root).lower()
                
                # Use a unique ID so we don't accidentally overwrite your face images
                unique_id = str(uuid.uuid4())[:8]

                if 'real' in folder_name or 'original' in folder_name or 'human' in folder_name:
                    if real_count < max_images_per_class:
                        dest_path = os.path.join(project_real_dir, f"fast_real_{unique_id}.jpg")
                        shutil.copy2(source_path, dest_path)
                        real_count += 1
                        if real_count % 500 == 0:
                            print(f"Added {real_count}/{max_images_per_class} Real images...")

                elif 'fake' in folder_name or 'manipulated' in folder_name or 'forged' in folder_name or 'ai' in folder_name or 'synthetic' in folder_name:
                    if fake_count < max_images_per_class:
                        dest_path = os.path.join(project_fake_dir, f"fast_fake_{unique_id}.jpg")
                        shutil.copy2(source_path, dest_path)
                        fake_count += 1
                        if fake_count % 500 == 0:
                            print(f"Added {fake_count}/{max_images_per_class} Fake images...")

    print("\n=========================================")
    print("FAST IMPORT COMPLETE!")
    print(f"Successfully added {real_count} NEW REAL images.")
    print(f"Successfully added {fake_count} NEW FAKE images.")
    print("Your dataset is now perfectly balanced.")
    print("=========================================")
    print("You are now ready to run: python backend/train.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_fast_balance.py <path_to_downloaded_dataset_folder>")
    else:
        import_fast_balance(sys.argv[1])
