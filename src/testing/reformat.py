import json
import os

def reformat_dataset(input_filename="metadata_part_000.jsonl", output_filename="train_dataset_reformatted.jsonl", start_index=6539):
    """
    Reads a dataset in jsonl format, reformats the audio_path keys,
    and saves the reformatted data to a new jsonl file.

    Args:
        input_filename (str): The name of the input jsonl file.
        output_filename (str): The name of the output jsonl file.
        start_index (int): The starting number for the new audio_path sequence.
                           The files will be named frag_XXXXX.wav where XXXXX
                           starts from this index, zero-padded to 5 digits.
    """
    current_index = start_index
    reformatted_data = []
    processed_count = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:

            for line in infile:
                try:
                    # Parse the JSON object from the line
                    data_entry = json.loads(line)

                    # Ensure 'text' key exists, as 'audio_path' will be replaced
                    if 'text' in data_entry:
                        # Generate the new audio path
                        # We need 5 digits, so format as {:05d}
                        new_audio_path = f"fragments/frag_{current_index:05d}.wav"

                        # Create the new dictionary with the reformatted path and original text
                        new_entry = {
                            "audio_path": new_audio_path,
                            "text": data_entry["text"]
                        }

                        # Write the new entry as a JSON line to the output file
                        outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')

                        current_index += 1
                        processed_count += 1
                    else:
                        print(f"Skipping line {current_index - start_index + 1}: 'text' key not found.")

                except json.JSONDecodeError:
                    print(f"Skipping line: Invalid JSON format.")
                except Exception as e:
                    print(f"An error occurred while processing line {current_index - start_index + 1}: {e}")


        print(f"Successfully reformatted {processed_count} entries.")
        print(f"Reformatted data saved to '{output_filename}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- How to run the script ---
if __name__ == "__main__":
    # Make sure your 'train_dataset.jsonl' file is in the same directory
    # as the script, or provide the full path.
    reformat_dataset()

    # If you want to start from a different index or use different filenames,
    # you can call the function like this:
    # reformat_dataset(input_filename="my_data.jsonl", output_filename="my_data_new.jsonl", start_index=1000)