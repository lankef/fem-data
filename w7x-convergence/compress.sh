#!/bin/bash

# Compress all folders starting with 'w7x_coil_0'
# Usage: ./compress_w7x_coil.sh [target_directory]
# If no target directory is provided, the current directory is used.

SEARCH_DIR="${1:-.}"

echo "Searching for folders starting with 'w7x_coil_0' in: $SEARCH_DIR"

found=0

for folder in "$SEARCH_DIR"/w7x_coil_0*/; do
    # Skip if no matching folders exist
    [ -d "$folder" ] || continue

    found=1
    folder_name=$(basename "$folder")
    archive="${folder_name}.tar.gz"

    echo "Compressing '$folder_name' -> '$archive' ..."
    tar -czf "$SEARCH_DIR/$archive" -C "$SEARCH_DIR" "$folder_name"

    if [ $? -eq 0 ]; then
        echo "  Done: $archive"
    else
        echo "  ERROR: Failed to compress '$folder_name'" >&2
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No folders starting with 'w7x_coil_0' found in '$SEARCH_DIR'."
    exit 1
fi

echo "All done."