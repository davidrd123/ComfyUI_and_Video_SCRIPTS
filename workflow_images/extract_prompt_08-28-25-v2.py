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

def extract_prompts(workflow_data):
    """
    Extracts both positive and negative prompts from workflow data.
    Handles both API format (flat dict) and UI format (nodes array).
    
    Args:
        workflow_data (dict): The parsed JSON workflow data.
        
    Returns:
        dict: {"positive": str|None, "negative": str|None}
    """
    if not isinstance(workflow_data, dict):
        print("Error: Invalid workflow data provided (not a dictionary).")
        return {"positive": None, "negative": None}
    
    # UI graph format (has "nodes" array)
    if "nodes" in workflow_data:
        nodes = {n["id"]: n for n in workflow_data["nodes"]}
        
        # 1) Try WanVideoTextEncode first
        wte = next((n for n in nodes.values() if n.get("type") == "WanVideoTextEncode"), None)
        if wte and isinstance(wte.get("widgets_values"), list):
            pos = wte["widgets_values"][0] if len(wte["widgets_values"]) > 0 else None
            neg = wte["widgets_values"][1] if len(wte["widgets_values"]) > 1 else None
            if pos and pos.strip():
                return {"positive": pos, "negative": neg}
        
        # 2) Try TextEncodeQwenImageEdit nodes with link-based detection
        qwen_nodes = [n for n in nodes.values() if n.get("type") == "TextEncodeQwenImageEdit"]
        if qwen_nodes and "links" in workflow_data:
            pos_prompt = None
            neg_prompt = None
            
            # Find sampler nodes to determine positive/negative connections
            sampler_nodes = [n for n in nodes.values() if "sampler" in n.get("type", "").lower()]
            
            for node in qwen_nodes:
                if node.get("widgets_values") and len(node["widgets_values"]) > 0:
                    prompt = node["widgets_values"][0]
                    if not prompt or not prompt.strip():
                        continue
                    
                    # Look for outgoing links from this node
                    outgoing_links = [l for l in workflow_data["links"] if l[1] == node["id"]]
                    
                    # Check if this node connects to a sampler's positive (slot 1) or negative (slot 2) input
                    is_positive = False
                    is_negative = False
                    
                    for link in outgoing_links:
                        # link format: [link_id, source_node_id, source_slot, target_node_id, target_slot, type]
                        target_node_id = link[3]
                        target_slot = link[4]
                        
                        # Check if target is a sampler
                        target_node = nodes.get(target_node_id)
                        if target_node and "sampler" in target_node.get("type", "").lower():
                            if target_slot == 1:  # Positive conditioning input
                                is_positive = True
                            elif target_slot == 2:  # Negative conditioning input
                                is_negative = True
                    
                    # Assign based on connection analysis
                    if is_positive and not pos_prompt:
                        pos_prompt = prompt
                    elif is_negative and not neg_prompt:
                        neg_prompt = prompt
                    elif not pos_prompt and not neg_prompt:
                        # Fallback: if no clear connection found, assume first non-empty is positive
                        pos_prompt = prompt
            
            if pos_prompt or neg_prompt:
                return {"positive": pos_prompt, "negative": neg_prompt}
        
        # 3) Fallback: CLIPTextEncode via TextEmbedBridge wiring
        bridge = next((n for n in nodes.values() if n.get("type") == "WanVideoTextEmbedBridge"), None)
        if bridge and "links" in workflow_data:
            # Find links: which node feeds bridge input slot 0 (positive) vs 1 (negative)
            pos_link = next((l for l in workflow_data["links"] if l[4] == bridge["id"] and l[5] == 0), None)
            neg_link = next((l for l in workflow_data["links"] if l[4] == bridge["id"] and l[5] == 1), None)
            
            def read_clip_encode_text(node_id):
                n = nodes.get(node_id)
                if n and n.get("type") == "CLIPTextEncode" and n.get("widgets_values"):
                    return n["widgets_values"][0]
                return None
            
            pos = read_clip_encode_text(pos_link[1]) if pos_link else None
            neg = read_clip_encode_text(neg_link[1]) if neg_link else None
            return {"positive": pos, "negative": neg}
        
        return {"positive": None, "negative": None}
    
    # API graph format (flat dictionary)
    else:
        for nid, node in workflow_data.items():
            if isinstance(node, dict) and node.get("class_type") == "WanVideoTextEncode":
                inputs = node.get("inputs", {})
                return {"positive": inputs.get("positive_prompt"), "negative": inputs.get("negative_prompt")}
        
        # Also check for TextEncodeQwenImageEdit in API format
        for nid, node in workflow_data.items():
            if isinstance(node, dict) and node.get("class_type") == "TextEncodeQwenImageEdit":
                inputs = node.get("inputs", {})
                if "prompt" in inputs:
                    return {"positive": inputs["prompt"], "negative": None}
        
        return {"positive": None, "negative": None}

def extract_specific_prompt_from_json_data(workflow_data):
    """
    Extracts the positive prompt from workflow data (maintains backward compatibility).
    
    Args:
        workflow_data (dict): The parsed JSON workflow data.

    Returns:
        str: The extracted positive prompt string, or None if not found.
    """
    prompts = extract_prompts(workflow_data)
    if prompts["positive"]:
        print(f"Found positive prompt")
        return prompts["positive"]
    elif prompts["negative"]:
        print(f"Found negative prompt (no positive found)")
        return prompts["negative"]
    else:
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
