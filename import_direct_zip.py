import os
import sys
import uuid
import zipfile

def extract_fast_balance(zip_file_path, max_images_per_class=2500):
    """
    This script bypasses the horribly slow Windows File Explorer extraction!
    It opens the massive zip file and instantly pulls out exactly the 5000 images we need
    directly into your Data folders, saving you hours of waiting.
    """
    project_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
    project_real_dir = os.path.join(project_data_dir, "REAL")
    project_fake_dir = os.path.join(project_data_dir, "FAKE")

    os.makedirs(project_real_dir, exist_ok=True)
    os.makedirs(project_fake_dir, exist_ok=True)

    print(f"Opening massive zip file: {zip_file_path}")
    print("Please wait, reading contents... (this takes a few seconds)")

    real_count = 0
    fake_count = 0

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Get a list of all files in the zip
            all_files = zip_ref.namelist()
            
            for file_path in all_files:
                if real_count >= max_images_per_class and fake_count >= max_images_per_class:
                    break # We have enough images!

                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    folder_name = file_path.lower()
                    unique_id = str(uuid.uuid4())[:8]

                    # Check if the path contains 'real' or 'fake' keywords
                    if 'real' in folder_name or 'original' in folder_name or 'human' in folder_name:
                        if real_count < max_images_per_class:
                            dest_path = os.path.join(project_real_dir, f"fast_zip_real_{unique_id}.jpg")
                            
                            # Extract directly to the destination
                            with zip_ref.open(file_path) as source, open(dest_path, "wb") as target:
                                target.write(source.read())
                            
                            real_count += 1
                            if real_count % 500 == 0:
                                print(f"Instantly extracted {real_count}/{max_images_per_class} Real objects...")

                    elif 'fake' in folder_name or 'manipulated' in folder_name or 'forged' in folder_name or 'ai' in folder_name or 'synthetic' in folder_name:
                        if fake_count < max_images_per_class:
                            dest_path = os.path.join(project_fake_dir, f"fast_zip_fake_{unique_id}.jpg")
                            
                            with zip_ref.open(file_path) as source, open(dest_path, "wb") as target:
                                target.write(source.read())
                                
                            fake_count += 1
                            if fake_count % 500 == 0:
                                print(f"Instantly extracted {fake_count}/{max_images_per_class} Fake objects...")

        print("\n=========================================")
        print("INSTANT ZIP IMPORT COMPLETE!")
        print(f"Successfully extracted {real_count} REAL images directly.")
        print(f"Successfully extracted {fake_count} FAKE images directly.")
        print("You just saved 3 hours! You can now cancel the Windows extraction.")
        print("=========================================")
        print("You are now ready to run: python backend/train.py")

    except Exception as e:
        print(f"Error reading zip file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_direct_zip.py <path_to_downloaded_zip_file>")
        print(r'Example: python import_direct_zip.py "C:\Users\Name\Downloads\archive.zip"')
    else:
        extract_fast_balance(sys.argv[1])
