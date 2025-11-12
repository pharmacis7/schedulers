import os

# Output file name
output_file = "all_code_output.txt"

# Open output file in write mode
with open(output_file, "w", encoding="utf-8") as outfile:
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                abs_path = os.path.abspath(file_path)
                
                # Write the file path
                outfile.write(f"{abs_path}\n")
                outfile.write("-" * len(abs_path) + "\n")
                
                # Try to read file contents
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        code = infile.read()
                        outfile.write(code)
                except Exception as e:
                    outfile.write(f"Error reading file: {e}")
                
                # Add spacing between files
                outfile.write("\n\n" + "="*80 + "\n\n")

print(f"✅ All Python code has been written to '{output_file}'.")
