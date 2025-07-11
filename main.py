from pre_processing import load_and_split_csv_files

chunks = load_and_split_csv_files()
print(f"Total chunks: {len(chunks)}")
