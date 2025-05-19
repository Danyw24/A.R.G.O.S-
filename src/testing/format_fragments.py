import os
import argparse
from pathlib import Path

def rename_wav_sequentially(
    target_directory: str,
    start_index: int,
    dry_run: bool = False
):
    """
    Renames all .wav files in the target directory sequentially.

    Files are renamed to 'fragment_XXXXX.wav' starting from the specified
    start_index, zero-padded to 5 digits. The renaming order is based on
    the alphabetical sorting of the original filenames.

    Args:
        target_directory (str): The path to the directory containing .wav files.
        start_index (int): The starting number for the sequential renaming.
        dry_run (bool): If True, print intended actions without renaming files.
    """
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"--- Sequential WAV Renaming ---")
    print(f"Target Directory: {target_directory}")
    print(f"Start Index: {start_index}")
    if dry_run:
        print("DRY RUN active: No files will be renamed.")
    print("-----------------------------")

    try:
        # Validate directory
        target_path = Path(target_directory)
        if not target_path.is_dir():
            print(f"Error: Target directory '{target_directory}' not found or is not a directory.")
            return

        # Validate start_index
        if start_index < 0:
            print(f"Error: Start index must be a non-negative integer.")
            return

        # Get all .wav files (case-insensitive) and sort them alphabetically
        wav_files = sorted([f for f in os.listdir(target_path) if f.lower().endswith('.wav')])

        if not wav_files:
            print("No .wav files found in the target directory.")
            return

        print(f"Found {len(wav_files)} .wav files to process.")

        current_index = start_index
        for old_filename in wav_files:
            old_filepath = target_path / old_filename
            new_filename = f"frag_{current_index:05d}.wav"
            new_filepath = target_path / new_filename

            # Check if the file already has the target name
            if old_filepath == new_filepath:
                print(f"Skipping: '{old_filename}' already has the correct name.")
                skipped_count += 1
                current_index += 1 # Still increment index to avoid reusing it
                continue

            # Check if the target filename already exists (collision)
            if new_filepath.exists():
                print(f"Error: Target filename '{new_filename}' already exists. Skipping rename for '{old_filename}'.")
                error_count += 1
                # Decide if you want to increment current_index here or not.
                # Incrementing avoids reusing the number but might create gaps.
                # Not incrementing might lead to repeated collision errors.
                # Let's increment to avoid repeated errors on the same target name.
                current_index += 1
                continue

            # Perform rename
            print(f"Renaming: '{old_filename}' -> '{new_filename}'")
            if not dry_run:
                try:
                    os.rename(old_filepath, new_filepath)
                    renamed_count += 1
                except OSError as e:
                    print(f"  Error renaming '{old_filename}': {e}")
                    error_count += 1
            else:
                 # In dry run, count it as if it were renamed for the summary
                 renamed_count += 1

            current_index += 1

        print("\n--- Renaming Summary ---")
        if dry_run:
             print(f"Files that would be renamed: {renamed_count}")
        else:
             print(f"Files successfully renamed: {renamed_count}")
        print(f"Files skipped (already correct or collision): {skipped_count + error_count}")
        print(f"  - Already correct name: {skipped_count}")
        print(f"  - Target name collision / Errors: {error_count}")
        print("------------------------")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- How to run the script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename .wav files in a directory sequentially.")
    parser.add_argument(
        "directory",
        help="Path to the directory containing the .wav files to rename."
    )
    parser.add_argument(
        "start_index",
        type=int,
        help="The starting index number for renaming (e.g., 2050)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the process without actually renaming any files."
    )

    args = parser.parse_args()

    rename_wav_sequentially(
        target_directory=args.directory,
        start_index=args.start_index,
        dry_run=args.dry_run
    )   

    # Example command-line usage:
    # python rename_simple.py /path/to/your/wav_folder 2050 --dry-run
    #
    # (Remove --dry-run to actually perform the renaming)
    # 5346