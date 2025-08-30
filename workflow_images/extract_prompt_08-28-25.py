import json
import os
import argparse
from PIL import Image, PngImagePlugin
import glob # For finding files in a directory

def get_workflow_from_image(image_path):
    """
    Extracts the workflow JSON string from a PNG image's metadata.

    Args:
        image_path (str): Path to the PNG image file.

    Returns:
        str: The workflow JSON string, or None if not found or on error.
    """
    try:
        img = Image.open(image_path)
        if img.info and isinstance(img.info, dict):
            # Common keys for ComfyUI workflow metadata
            for key in ["workflow", "prompt"]:
                if key in img.info:
                    return img.info[key]
        # print(f"No workflow metadata found in {image_path}") # Less verbose for batch processing
        return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error reading image or its metadata {image_path}: {e}")
        return None

def extract_specific_prompt_from_json_data(workflow_data):
    """
    Extracts prompts from supported node types from parsed workflow JSON data.
    Supports both ComfyUI node format and flat dictionary format.

    Args:
        workflow_data (dict): The parsed JSON workflow data.

    Returns:
        str: The extracted prompt string, or None if not found.
    """
    if not isinstance(workflow_data, dict):
        print("Error: Invalid workflow data provided (not a dictionary).")
        return None

    # Define supported node types that contain prompts
    supported_node_types = ["WanVideoTextEncode", "TextEncodeQwenImageEdit"]
    
    # Store all found prompts with their colors to prioritize green (positive) prompts
    green_prompts = []  # Green nodes (positive prompts)
    other_prompts = []  # Non-green nodes (negative prompts, etc.)
    
    # Check if this is the ComfyUI nodes format (has "nodes" array)
    if "nodes" in workflow_data:
        for node in workflow_data.get("nodes", []):
            node_type = node.get("type")
            if node_type in supported_node_types:
                widgets_values = node.get("widgets_values")
                node_color = node.get("color", "").lower()  # Get color, default to empty string
                
                if widgets_values and isinstance(widgets_values, list) and len(widgets_values) > 0:
                    prompt_text = widgets_values[0]
                    if prompt_text and prompt_text.strip():  # Non-empty prompt
                        # Check if this is a green node (positive prompt)
                        # #232 = positive (green-ish), #223 = negative (red-ish)
                        if "#232" in node_color or "green" in node_color:
                            green_prompts.append((node_type, prompt_text, node_color))
                        else:
                            other_prompts.append((node_type, prompt_text, node_color))
    
    # Check if this is the flat dictionary format (node IDs as keys)
    else:
        for node_id, node_data in workflow_data.items():
            if isinstance(node_data, dict):
                node_type = node_data.get("class_type")
                if node_type in supported_node_types:
                    inputs = node_data.get("inputs", {})
                    
                    # For WanVideoTextEncode, look for positive_prompt and negative_prompt
                    if "positive_prompt" in inputs:
                        prompt_text = inputs["positive_prompt"]
                        if prompt_text and prompt_text.strip():
                            green_prompts.append((node_type, prompt_text, "positive"))
                    
                    if "negative_prompt" in inputs:
                        prompt_text = inputs["negative_prompt"]
                        if prompt_text and prompt_text.strip():
                            other_prompts.append((node_type, prompt_text, "negative"))
    
    # Prioritize green prompts (positive), then fall back to others
    if green_prompts:
        node_type, prompt, color = green_prompts[0]
        print(f"Found prompt in {node_type} node (positive)")
        return prompt
    elif other_prompts:
        node_type, prompt, color = other_prompts[0]
        print(f"Found prompt in {node_type} node")
        return prompt
    
    # print(f"Error: None of the supported node types {supported_node_types} found in the workflow data.") # Less verbose
    return None

def process_png_file(input_png_file, output_directory, extract_json=False):
    """Processes a single PNG file to extract and save the specific prompt or full workflow JSON.
    Only creates the output file if it doesn't already exist.
    
    Args:
        input_png_file (str): Path to the PNG file to process
        output_directory (str): Directory to save the output file
        extract_json (bool): If True, extract full workflow JSON; if False, extract prompt text
    """
    base_filename = os.path.splitext(os.path.basename(input_png_file))[0]
    
    if extract_json:
        output_file = os.path.join(output_directory, f"{base_filename}.json")
        file_type = "workflow JSON"
    else:
        output_file = os.path.join(output_directory, f"{base_filename}.txt")
        file_type = "prompt"

    if os.path.exists(output_file):
        print(f"Skipping {input_png_file}, {file_type} file {output_file} already exists.")
        return

    print(f"Processing {input_png_file}...")
    workflow_json_string = get_workflow_from_image(input_png_file)

    if workflow_json_string:
        try:
            workflow_json_data = json.loads(workflow_json_string)
            
            if extract_json:
                # Save the entire workflow JSON with pretty formatting
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(workflow_json_data, f, indent=2, ensure_ascii=False)
                    print(f"Successfully created workflow JSON file {output_file}")
                    
                    # Set timestamp from source PNG
                    try:
                        source_png_stat = os.stat(input_png_file)
                        source_png_mtime = source_png_stat.st_mtime
                        source_png_atime = source_png_stat.st_atime
                        os.utime(output_file, (source_png_atime, source_png_mtime))
                        print(f"Updated timestamp of {output_file} to match {input_png_file}")
                    except OSError as e:
                        if e.errno == 1: # Operation not permitted
                            print(f"Warning: Could not set timestamp for {output_file} (Operation not permitted). File created with current timestamp.")
                        else:
                            print(f"Warning: Error setting timestamp for {output_file}: {e}")
                            
                except IOError as e:
                    print(f"Error: Could not write to output file {output_file}. {e}")
            else:
                # Extract specific prompt (original behavior)
                specific_prompt = extract_specific_prompt_from_json_data(workflow_json_data)

                if specific_prompt:
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(specific_prompt)
                        print(f"Successfully created prompt file {output_file}")

                        try:
                            # Attempt to set timestamp from source PNG
                            source_png_stat = os.stat(input_png_file)
                            source_png_mtime = source_png_stat.st_mtime
                            source_png_atime = source_png_stat.st_atime
                            os.utime(output_file, (source_png_atime, source_png_mtime))
                            print(f"Updated timestamp of {output_file} to match {input_png_file}")
                        except OSError as e:
                            if e.errno == 1: # Operation not permitted
                                print(f"Warning: Could not set timestamp for {output_file} (Operation not permitted). File created with current timestamp.")
                            else:
                                print(f"Warning: Error setting timestamp for {output_file}: {e}")                            

                    except IOError as e:
                        print(f"Error: Could not write to output file {output_file}. {e}")
                else:
                    # If specific_prompt is None, it means none of the supported node types or their values weren't found.
                    # We might not want to create an empty .txt file, or we might. 
                    # For now, we only create a file if a prompt is found.
                    print(f"No specific prompt found in {input_png_file}. No output file created.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode workflow JSON from {input_png_file}.")
    else:
        print(f"Failed to get workflow JSON from {input_png_file}. No output file created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract prompts or full workflow JSON from PNG images with embedded ComfyUI workflow, from a single file or a directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", dest="png_file", help="Path to a single PNG image file with embedded workflow.")
    group.add_argument("-d", "--directory", dest="png_directory", help="Path to a directory containing PNG image files.")
    
    parser.add_argument("-o", "--output_dir", help="Directory to save the extracted files. If not specified, output is saved alongside source files.")
    parser.add_argument("-j", "--json", action="store_true", help="Extract full workflow JSON instead of just the prompt text. Creates .json files instead of .txt files.")

    args = parser.parse_args()

    determined_output_dir = None

    if args.output_dir:
        determined_output_dir = os.path.abspath(args.output_dir)
        if not os.path.isdir(determined_output_dir):
            try:
                os.makedirs(determined_output_dir, exist_ok=True)
                print(f"Created output directory: {determined_output_dir}")
            except OSError as e:
                print(f"Error: Could not create specified output directory {determined_output_dir}. {e}")
                exit(1)
    elif args.png_file: # Default for single file: output alongside the file
        determined_output_dir = os.path.dirname(os.path.abspath(args.png_file))
    elif args.png_directory: # Default for directory: output into the source directory
        determined_output_dir = os.path.abspath(args.png_directory)
    
    # This check is crucial to ensure determined_output_dir is valid before proceeding
    if not determined_output_dir or not os.path.isdir(determined_output_dir):
        print(f"Error: Output directory '{determined_output_dir}' is not valid or could not be determined. Please check paths.")
        exit(1)

    if args.png_file:
        if not os.path.isfile(args.png_file):
            print(f"Error: File not found at {args.png_file}")
            exit(1)
        process_png_file(args.png_file, determined_output_dir, args.json)
    elif args.png_directory:
        if not os.path.isdir(args.png_directory):
            print(f"Error: Directory not found at {args.png_directory}")
            exit(1)
        
        search_pattern = os.path.join(args.png_directory, '*.png')
        png_files = glob.glob(search_pattern)
        if not png_files:
            print(f"No PNG files found in {args.png_directory} (using pattern: {search_pattern})")
        else:
            print(f"Found {len(png_files)} PNG files in {args.png_directory}.")
            for png_file_path in png_files:
                process_png_file(png_file_path, determined_output_dir, args.json)
                print("---") # Separator for multiple files
    
    print("Processing complete.")

# Example of how you might batch process (conceptual):
# import glob
# for png_image_path in glob.glob("path/to/your/images/*.png"):
#     # Call the main logic here, perhaps refactored into a function
#     print(f"Processing {png_image_path}...")
#     # workflow_json_string = get_workflow_from_image(png_image_path)
#     # ... and so on ...
