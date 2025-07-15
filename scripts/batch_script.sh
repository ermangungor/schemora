#!/bin/bash
set -e

# Specify the directory to search
directory_path="/Users/ogungor/Documents/schemora/scripts/omop_configs"
#directory_path="/Users/ogungor/Documents/schemora/scripts/synt_configs"

# Iterate over all .toml files using glob
for toml_file in "$directory_path"/*.toml; do
  # Check if the file exists (in case there are no .toml files)
  if [ -e "$toml_file" ]; then
    echo "Processing file: $toml_file"
    make CONFIG_FILE="$toml_file" all

    # Add your processing logic here
    # For example, you can print the contents or process them further using other commands
  else
    echo "No TOML files found in directory."
  fi
done