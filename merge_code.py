import os

# Files or folders to ignore
IGNORE_LIST = {'node_modules', '.git', 'venv', '__pycache__', 'target', 'dist', '.gitignore', 'Test Results.txt', 'Test Architecture.md', 'Commands.md', 'microservices-demo', 'Chekpoint File', 'Details.md', 'merge_code.py', '.pytest_cache'}
# File extensions to include (Note: added '.' to md for consistency)
EXTENSIONS = {'.py', '.js', '.java', '.go', '.ts', '.yaml', '.yml', '.proto', '.md'}

def merge_files(output_filename="project_context.txt"):
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk('.'):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_LIST]
            
            for file in files:
                # Check if file is in ignore list or has a blocked extension
                if file in IGNORE_LIST:
                    continue
                    
                if any(file.endswith(ext) for ext in EXTENSIONS):
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write(f"{'='*50}\n\n") # Fixed the syntax error here
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Could not read file: {e}\n")
    print(f"Done! All code merged into {output_filename}")

if __name__ == "__main__":
    merge_files()