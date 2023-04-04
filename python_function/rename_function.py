import os

name = 'Trump'
# Set the directory containing the pictures
dir_path = f'../image_train/{name}'

# Get a list of all the files in the directory
file_list = os.listdir(dir_path)

# Start to rename the picture from picture_0001.jpg
i = 1

# Loop through the files and rename them
for i, file_name in enumerate(file_list):
    # Construct the new file name
    new_file_name = f'{name}_000{i}.jpg'

    # Construct the full file paths for the old and new names
    old_path = os.path.join(dir_path, file_name)
    new_path = os.path.join(dir_path, new_file_name)

    # Rename the file
    os.rename(old_path, new_path)

    # Print a message indicating the file has been renamed
    print('Renamed {} to {}'.format(file_name, new_file_name))
