import os

def write_file_structure_to_text(root_dir, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(root_dir):
            # Calculate the relative level in the directory structure
            level = root.replace(root_dir, '').count(os.sep)
            indent = '    ' * level  # Indentation for folder names

            # Write the folder name
            folder_name = os.path.basename(root)
            f.write(f"{indent}📁{folder_name}\n")

            # Write the files in the current directory
            sub_indent = '    ' * (level + 1)  # Indentation for file names
            for file_name in files:
                f.write(f"{sub_indent}└── {file_name}\n")
                
                # Write the content of each file
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r') as file:
                        file_content = file.read()
                        file_content = file_content.replace('\n', f'\n{sub_indent}    ')
                        f.write(f"{sub_indent}    Content:\n{sub_indent}    {file_content}\n\n")
                except Exception as e:
                    f.write(f"{sub_indent}    [Error reading file: {e}]\n\n")

# Specify the root directory and the output file
root_directory = '/home/harikrishnan/ROS_PROJECTS/snake_robot_harmonic_ws/src'  # Change to your root directory
output_filename = 'directory_structure.txt'

write_file_structure_to_text(root_directory, output_filename)
print(f"Directory structure and file contents have been written to '{output_filename}'.")
