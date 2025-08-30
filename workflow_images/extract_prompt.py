import json
import os
import argparse
import logging
from PIL import Image, PngImagePlugin
import glob # For finding files in a directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _safe_get(lst, idx, default=None):
    """Safe list access with bounds checking."""
    return lst[idx] if isinstance(lst, list) and len(lst) > idx else default

def _norm(s):
    """Normalize text: strip whitespace, return None if empty."""
    return (s or "").strip() or None

class UIGraph:
    """
    Normalizes a ComfyUI editor graph (nodes + links) for robust querying.
    
    Mental Model: ComfyUI workflows come in two shapes:
    
    1. API graph (flat dict): {"16": {"class_type": "...", "inputs": {...}}, ...}
       - Prompts live in 'inputs' (e.g., WanVideoTextEncode.inputs.positive_prompt)
       - What you POST to /prompt endpoint
       
    2. UI graph (nodes + links): {"nodes": [...], "links": [...]}
       - Text stored in 'widgets_values', wiring determines positive vs negative
       - What the editor saves in PNG metadata
    
    This class handles UI graphs by providing three key primitives:
    1. Find samplers that declare 'positive'/'negative' inputs
    2. Map input names → slot indices (no hardcoded slot assumptions)  
    3. Follow links to source nodes, then read text per encoder type
    
    The pattern: name → slot → link → source node → per-type reader
    """
    
    # Whitelist of known sampler types - safer than blacklisting bridge nodes
    KNOWN_POSNEG_SAMPLERS = ["KSampler", "ClownsharKSampler_Beta"]
    KNOWN_TEXT_EMBEDS_SAMPLERS = ["WanVideoSampler"]
    
    # Output node types that indicate a sampler is part of the active workflow
    OUTPUT_TYPES = {"VHS_VideoCombine", "SaveImage", "WanVideoDecode"}
    
    # Extensibility: Centralized encoder text readers
    ENCODER_READERS = {
        "CLIPTextEncode": lambda n, w, which: _norm(_safe_get(w, 0)),
        "TextEncodeQwenImageEdit": lambda n, w, which: _norm(_safe_get(w, 0)),
        "WanVideoTextEncode": lambda n, w, which: _norm(_safe_get(w, 1) if which == "negative" else _safe_get(w, 0)),
        # Extension point: Add new encoders here
        # "SomeFutureEncoder": lambda n, w, which: _norm(_safe_get(w, 0)),
    }

    def __init__(self, wf):
        # 1. Create fast node lookup: node_id -> node_object
        self.nodes = {n["id"]: n for n in wf.get("nodes", [])}
        # Example from your JSON:
        # self.nodes[76] = {"id": 76, "type": "TextEncodeQwenImageEdit", 
        #                   "widgets_values": ["Perform a photorealistic edit..."], 
        #                   "color": "#232"}
        # self.nodes[77] = {"id": 77, "type": "TextEncodeQwenImageEdit", 
        #                   "widgets_values": ["blurry, ugly"], "color": "#223"}
        # self.nodes[101] = {"id": 101, "type": "ClownsharKSampler_Beta", 
        #                    "inputs": [{"name": "positive"}, {"name": "negative"}, ...]}
        
        # 2. Store links array for reference
        # links format: [link_id, src_node_id, src_slot, dst_node_id, dst_slot, data_type]
        self.links = wf.get("links", [])
        # Examples from your JSON:
        # [189, 77, 0, 101, 2, "CONDITIONING"]  # Node 77 -> Node 101 slot 2
        # [194, 76, 0, 101, 1, "CONDITIONING"]  # Node 76 -> Node 101 slot 1
        
        # 3. BUILD THE MAGIC: Reverse lookup index
        # Maps (destination_node_id, input_slot) -> source_node_id
        self.in_edge = {}
        for L in self.links:
            try:
                # From your basketball JSON:
                # L = [189, 77, 0, 101, 2, "CONDITIONING"]
                #      [id, src, src_slot, dst, dst_slot, type]
                src, dst, dst_slot = L[1], L[3], L[4]
                #     src=77,    dst=101, dst_slot=2
                
                # Creates: self.in_edge[(101, 2)] = 77
                # Meaning: "Node 101's input slot 2 is fed by Node 77"
                self.in_edge[(dst, dst_slot)] = src
                
                # After processing your two key links:
                # self.in_edge[(101, 1)] = 76  # Positive input (slot 1) <- Node 76 (long prompt)
                # self.in_edge[(101, 2)] = 77  # Negative input (slot 2) <- Node 77 ("blurry, ugly")
            except Exception:
                pass  # Skip malformed links

    def find_nodes(self, types):
        """
        Find all nodes matching the specified type(s).
        
        Args:
            types: String or list of node type names to search for
            
        Returns:
            List of matching node objects
            
        Example:
            samplers = g.find_nodes(["KSampler", "ClownsharKSampler_Beta"])
            encoders = g.find_nodes("TextEncodeQwenImageEdit")
        """
        if isinstance(types, str): 
            types = (types,)
        tset = set(types)  # O(1) membership testing
        return [n for n in self.nodes.values() if n.get("type") in tset]

    def first_node(self, types):
        """
        Find first node matching the specified type(s).
        
        Args:
            types: String or list of node type names
            
        Returns:
            First matching node object, or None if none found
        """
        nodes = self.find_nodes(types)
        return nodes[0] if nodes else None

    def sampler_with_posneg(self):
        """
        Find known sampler node that has 'positive' and 'negative' inputs.
        
        Uses whitelist of known sampler types to avoid confusion with bridge nodes
        that also have positive/negative inputs. Prefers samplers that reach output nodes.
        
        Returns:
            Sampler node object, or None if not found
            
        Example from basketball JSON:
            Returns Node 101 (ClownsharKSampler_Beta) which has inputs:
            [{"name": "model"}, {"name": "positive"}, {"name": "negative"}, ...]
        """
        candidates = []
        for n in self.nodes.values():
            node_type = n.get("type")
            
            # WHITELIST: Only check known sampler types
            if node_type not in self.KNOWN_POSNEG_SAMPLERS:
                logger.debug(f"Skipping node {n['id']} ({node_type}) - not in known pos/neg samplers")
                continue
                
            # Check if this node has inputs with the right names
            ins = n.get("inputs") or []
            names = [i.get("name") for i in ins]
            if "positive" in names and "negative" in names:
                # CHECK: Are these inputs actually wired?
                pos_idx = self.input_index(n, "positive")
                neg_idx = self.input_index(n, "negative")
                pos_wired = self.src_for(n["id"], pos_idx) is not None
                neg_wired = self.src_for(n["id"], neg_idx) is not None
 
                logger.info(f"Found known sampler {n['id']} ({node_type}): positive wired={pos_wired}, negative wired={neg_wired}")
                
                if pos_wired or neg_wired:  # At least one input is wired
                    candidates.append(n)
        
        # Prefer samplers that reach output nodes
        for candidate in candidates:
            if self._reaches_output(candidate["id"]):
                logger.info(f"Selected sampler {candidate['id']} - reaches output")
                return candidate
        
        # Fall back to first candidate if none reach output
        if candidates:
            logger.info(f"Selected first sampler {candidates[0]['id']} - no output reach found")
            return candidates[0]
        
        return None
    
    def sampler_with_text_embeds(self):
        """
        Find known sampler node that has 'text_embeds' input AND verify it's wired.
        
        Uses whitelist of known text_embeds sampler types for safety.
        Prefers samplers that reach output nodes.

        Returns:
            Sampler node object, or None if not found
            
        Example from Graffito JSON:
            Returns Node 90 (WanVideoSampler) which has inputs:
            [{"name": "model"}, {"name": "image_embeds"}, {"name": "text_embeds"}, {"name": "samples"}, {"name": "feta_args"}, ...]
        """
        candidates = []
        for n in self.nodes.values():
            node_type = n.get("type")
            
            # WHITELIST: Only check known text_embeds sampler types
            if node_type not in self.KNOWN_TEXT_EMBEDS_SAMPLERS:
                logger.debug(f"Skipping node {n['id']} ({node_type}) - not in known text_embeds samplers")
                continue
                
            ins = n.get("inputs") or []
            names = [i.get("name") for i in ins]
            if "text_embeds" in names:
                # CHECK: Is text_embeds input actually wired?
                embeds_idx = self.input_index(n, "text_embeds")
                embeds_wired = self.src_for(n["id"], embeds_idx) is not None
                
                logger.info(f"Found known text_embeds sampler {n['id']} ({node_type}): text_embeds wired={embeds_wired}")
                
                if embeds_wired:
                    candidates.append(n)
        
        # Prefer samplers that reach output nodes
        for candidate in candidates:
            if self._reaches_output(candidate["id"]):
                logger.info(f"Selected text_embeds sampler {candidate['id']} - reaches output")
                return candidate
        
        # Fall back to first candidate if none reach output
        if candidates:
            logger.info(f"Selected first text_embeds sampler {candidates[0]['id']} - no output reach found")
            return candidates[0]
        
        return None

    def input_index(self, node, name):
        """
        Find which slot index corresponds to a named input.
        
        This replaces hardcoded slot assumptions (slot 1 = positive, etc.)
        with actual name-based lookup.
        
        Args:
            node: Node object to examine
            name: Input name to find ("positive", "negative", etc.)
            
        Returns:
            Slot index (int), or None if not found
            
        Example from basketball JSON Node 101:
            input_index(node_101, "positive") -> 1
            input_index(node_101, "negative") -> 2
        """
        for idx, inp in enumerate(node.get("inputs") or []):
            if inp.get("name") == name:
                return idx
        return None

    def src_for(self, dst_id, dst_slot):
        """
        Find what node feeds a specific input slot.
        
        This uses the pre-built reverse index for O(1) lookup instead of
        scanning all links repeatedly.
        
        Args:
            dst_id: Destination node ID
            dst_slot: Input slot index
            
        Returns:
            Source node ID, or None if no connection
            
        Examples from basketball JSON:
            src_for(101, 1) -> 76  # What feeds sampler's positive? Node 76
            src_for(101, 2) -> 77  # What feeds sampler's negative? Node 77
        """
        return self.in_edge.get((dst_id, dst_slot))

    def read_text_from_node(self, nid, which=None):
        """
        Read prompt text from known encoder node types.
        
        Uses centralized ENCODER_READERS table for extensibility.
        
        Args:
            nid: Node ID to read from
            which: 'positive' or 'negative' hint for dual-slot encoders
            
        Returns:
            Text string, or None if not found/empty
        """
        if nid is None: 
            return None
        n = self.nodes.get(nid) or {}
        node_type = n.get("type")
        widgets = n.get("widgets_values")

        # Use centralized encoder readers table
        reader = self.ENCODER_READERS.get(node_type)
        if reader:
            return reader(n, widgets, which)

        # Extension point: Add new encoder types to ENCODER_READERS table
        logger.debug(f"No reader found for encoder type: {node_type}")
        return None

    def _reaches_output(self, start_id, max_hops=200):
        """
        Check if a node has a path to an output node (save/decode).
        
        Uses downstream BFS to find if there's a path from start_id to any OUTPUT_TYPES node.
        
        Args:
            start_id: Node ID to start from
            max_hops: Maximum hops to search (prevents infinite loops)
            
        Returns:
            True if path exists to output node, False otherwise
        """
        # Build outgoing links index: src_id -> [dst_ids]
        out_by_src = {}
        for L in self.links:
            out_by_src.setdefault(L[1], []).append(L[3])
        
        # BFS from start_id
        seen, q = set([start_id]), [start_id]
        hops = 0
        while q and hops < max_hops:
            nxt = []
            for nid in q:
                for dst in out_by_src.get(nid, []):
                    if dst in seen: 
                        continue
                    if (self.nodes.get(dst) or {}).get("type") in self.OUTPUT_TYPES:
                        return True
                    seen.add(dst)
                    nxt.append(dst)
            q = nxt
            hops += 1
        return False

def get_workflow_from_image(image_path):
    """
    Extracts the workflow JSON string from a PNG image's metadata.

    Args:
        image_path (str): Path to the PNG image file.

    Returns:
        str: The workflow JSON string, or None if not found or on error.
    """
    try:
        with Image.open(image_path) as img:
            if img.info and isinstance(img.info, dict):
                # Common keys for ComfyUI workflow metadata
                for key in ["workflow", "prompt"]:
                    if key in img.info:
                        return img.info[key]
            logger.debug(f"No workflow metadata found in {image_path}")
            return None
    except FileNotFoundError:
        logger.error(f"Image file not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading image or its metadata {image_path}: {e}")
        return None

def extract_prompts(workflow):
    """
    Extracts both positive and negative prompts from workflow data.
    Uses robust wiring-based detection instead of brittle heuristics.
    
    Decision Tree:
    
    UI Graph Path (has "nodes" array):
    (a) Sampler wiring (preferred) - Follow name → slot → link → source pattern
        - Find sampler with "positive"/"negative" inputs (avoids slot assumptions)
        - Map input names to actual slot indices
        - Follow links to source encoder nodes  
        - Read text using per-type rules
        - Works for: Qwen Image Edit, various samplers
        
    (b) Bridge wiring (fallback) - CLIPTextEncode → WanVideoTextEmbedBridge
        - Bridge has consistent slot 0=positive, 1=negative
        - Follow links to CLIP encoder sources
        - Works for: Graffito-style workflows
        
    (c) Single encoder (fallback) - WanVideoTextEncode with both prompts
        - Text stored in widgets_values[0]=pos, [1]=neg
        - Works for: Simple Wan workflows
    
    API Graph Path (flat dict):
    - WanVideoTextEncode.inputs.positive_prompt/negative_prompt
    - TextEncodeQwenImageEdit.inputs.prompt (single prompt)
    
    Args:
        workflow (dict): The parsed JSON workflow data.
        
    Returns:
        dict: {"positive": str|None, "negative": str|None}
        
    Examples:
        Basketball JSON -> {"positive": "Perform a photorealistic edit...", 
                           "negative": "blurry, ugly"}
        Levetate JSON -> {"positive": API prompt, "negative": API negative}
        Graffito JSON -> {"positive": CLIP encoder text, "negative": CLIP encoder text}
    """
    if not isinstance(workflow, dict):
        print("Error: Invalid workflow data provided (not a dictionary).")
        return {"positive": None, "negative": None}
    
    # UI graph format (has "nodes" array)
    if "nodes" in workflow:
        logger.info("Processing UI graph format (nodes + links)")
        g = UIGraph(workflow)

        # (a) Sampler with pos/neg inputs (Qwen-edit, KSampler, ClownsharKSampler_Beta)
        sampler = g.sampler_with_posneg()
        if sampler:
            logger.info(f"Found pos/neg sampler node: {sampler['id']}")
            pos_idx = g.input_index(sampler, "positive")
            neg_idx = g.input_index(sampler, "negative")
            pos_src = g.src_for(sampler["id"], pos_idx) if pos_idx is not None else None
            neg_src = g.src_for(sampler["id"], neg_idx) if neg_idx is not None else None
            logger.info(f"Found positive source node: {pos_src}")
            logger.info(f"Found negative source node: {neg_src}")
            pos = g.read_text_from_node(pos_src, "positive")
            neg = g.read_text_from_node(neg_src, "negative")

            if pos or neg:
                return {"positive": pos, "negative": neg}
        
        # (b) Sampler with text_embeds (WanVideoSampler)
        sampler = g.sampler_with_text_embeds()
        if sampler:
            logger.info(f"Found text_embeds sampler node: {sampler['id']}")
            embeds_idx = g.input_index(sampler, "text_embeds")
            embeds_src = g.src_for(sampler["id"], embeds_idx) if embeds_idx is not None else None
            
            if embeds_src:
                logger.info(f"Found text_embeds source node: {embeds_src}")
                src_node = g.nodes.get(embeds_src, {})
                src_type = src_node.get("type")
                
                if src_type == "WanVideoTextEmbedBridge":
                    # Bridge has named inputs "positive"/"negative" → follow by name
                    logger.info(f"Tracing bridge node {embeds_src} for pos/neg inputs")
                    p_idx = g.input_index(src_node, "positive")
                    n_idx = g.input_index(src_node, "negative")
                    p_src = g.src_for(src_node["id"], p_idx) if p_idx is not None else None
                    n_src = g.src_for(src_node["id"], n_idx) if n_idx is not None else None
                    pos = g.read_text_from_node(p_src, "positive")
                    neg = g.read_text_from_node(n_src, "negative")
                    if pos or neg:
                        return {"positive": pos, "negative": neg}
                        
                elif src_type == "WanVideoTextEncode":
                    # WTE packs both prompts: widgets_values[0]=pos, [1]=neg
                    logger.info(f"Reading from WanVideoTextEncode node {embeds_src}")
                    w = src_node.get("widgets_values") or []
                    return {"positive": _safe_get(w, 0), "negative": _safe_get(w, 1)}
            
        # (c) Fallback: standalone WanVideoTextEncode in the graph
        wte = g.first_node("WanVideoTextEncode")
        if wte:
            logger.info(f"Found standalone WanVideoTextEncode node: {wte['id']}")
            w = wte.get("widgets_values") or []
            return {"positive": _safe_get(w, 0), "negative": _safe_get(w, 1)}

        return {"positive": None, "negative": None}
    
    # API graph format (flat dictionary)
    else:
        logger.info("Processing API graph format (flat dictionary)")
        for nid, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "WanVideoTextEncode":
                inputs = node.get("inputs", {})
                logger.info(f"Found WanVideoTextEncode node in API format: {nid}")
                return {"positive": inputs.get("positive_prompt"), "negative": inputs.get("negative_prompt")}
        
        # Also check for TextEncodeQwenImageEdit in API format
        for nid, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "TextEncodeQwenImageEdit":
                inputs = node.get("inputs", {})
                logger.info(f"Found TextEncodeQwenImageEdit node in API format: {nid}")
                if "prompt" in inputs:
                    return {"positive": inputs["prompt"], "negative": None}
        
        return {"positive": None, "negative": None}

def extract_specific_prompt_from_json_data(workflow):
    """
    Extracts the positive prompt from workflow data (maintains backward compatibility).
    
    Args:
        workflow_data (dict): The parsed JSON workflow data.

    Returns:
        str: The extracted positive prompt string, or None if not found.
    """
    prompts = extract_prompts(workflow)
    if prompts["positive"]:
        print(f"Found positive prompt")
        return prompts["positive"]
    elif prompts["negative"]:
        print(f"Found negative prompt (no positive found)")
        return prompts["negative"]
    else:
        return None

def process_png_file(input_png_file, output_directory, extract_json=False, extract_both=False, force_overwrite=False):
    """Processes a single PNG file to extract and save the specific prompt or full workflow JSON.
    
    Args:
        input_png_file (str): Path to the PNG file to process
        output_directory (str): Directory to save the output file
        extract_json (bool): If True, extract full workflow JSON; if False, extract prompt text
        extract_both (bool): If True, extract both positive/negative prompts as JSON
        force_overwrite (bool): If True, overwrite existing files; if False, skip existing files
    """
    base_filename = os.path.splitext(os.path.basename(input_png_file))[0]
    
    if extract_json:
        output_file = os.path.join(output_directory, f"{base_filename}.json")
        file_type = "workflow JSON"
    elif extract_both:
        output_file = os.path.join(output_directory, f"{base_filename}.json")
        file_type = "prompts JSON"
    else:
        output_file = os.path.join(output_directory, f"{base_filename}.txt")
        file_type = "prompt"

    if os.path.exists(output_file) and not force_overwrite:
        logger.info(f"Skipping {input_png_file}, {file_type} file {output_file} already exists. Use --force to overwrite.")
        return
    elif os.path.exists(output_file) and force_overwrite:
        logger.info(f"Overwriting existing {file_type} file {output_file}.")

    logger.info(f"Processing {input_png_file}...")
    workflow_json_string = get_workflow_from_image(input_png_file)

    if workflow_json_string:
        try:
            workflow_json_data = json.loads(workflow_json_string)
            
            if extract_json:
                # Save the entire workflow JSON with pretty formatting
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(workflow_json_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Successfully created workflow JSON file {output_file}")
                    
                    # Set timestamp from source PNG
                    try:
                        source_png_stat = os.stat(input_png_file)
                        source_png_mtime = source_png_stat.st_mtime
                        source_png_atime = source_png_stat.st_atime
                        os.utime(output_file, (source_png_atime, source_png_mtime))
                        logger.info(f"Updated timestamp of {output_file} to match {input_png_file}")
                    except OSError as e:
                        if e.errno == 1: # Operation not permitted
                            logger.warning(f"Could not set timestamp for {output_file} (Operation not permitted). File created with current timestamp.")
                        else:
                            logger.warning(f"Error setting timestamp for {output_file}: {e}")
                            
                except IOError as e:
                    logger.error(f"Could not write to output file {output_file}. {e}")
            elif extract_both:
                # Extract both positive and negative prompts as JSON
                prompts = extract_prompts(workflow_json_data)
                out = {"positive": prompts["positive"], "negative": prompts["negative"]}
                
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(out, f, indent=2, ensure_ascii=False)
                    logger.info(f"Successfully created prompts JSON file {output_file}")
                    
                    # Set timestamp from source PNG
                    try:
                        source_png_stat = os.stat(input_png_file)
                        source_png_mtime = source_png_stat.st_mtime
                        source_png_atime = source_png_stat.st_atime
                        os.utime(output_file, (source_png_atime, source_png_mtime))
                        logger.info(f"Updated timestamp of {output_file} to match {input_png_file}")
                    except OSError as e:
                        if e.errno == 1: # Operation not permitted
                            logger.warning(f"Could not set timestamp for {output_file} (Operation not permitted). File created with current timestamp.")
                        else:
                            logger.warning(f"Error setting timestamp for {output_file}: {e}")
                            
                except IOError as e:
                    logger.error(f"Could not write to output file {output_file}. {e}")
            else:
                # Extract specific prompt (original behavior)
                specific_prompt = extract_specific_prompt_from_json_data(workflow_json_data)

                if specific_prompt:
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(specific_prompt)
                        logger.info(f"Successfully created prompt file {output_file}")

                        try:
                            # Attempt to set timestamp from source PNG
                            source_png_stat = os.stat(input_png_file)
                            source_png_mtime = source_png_stat.st_mtime
                            source_png_atime = source_png_stat.st_atime
                            os.utime(output_file, (source_png_atime, source_png_mtime))
                            logger.info(f"Updated timestamp of {output_file} to match {input_png_file}")
                        except OSError as e:
                            if e.errno == 1: # Operation not permitted
                                logger.warning(f"Could not set timestamp for {output_file} (Operation not permitted). File created with current timestamp.")
                            else:
                                logger.warning(f"Error setting timestamp for {output_file}: {e}")                            

                    except IOError as e:
                        logger.error(f"Could not write to output file {output_file}. {e}")
                else:
                    # If specific_prompt is None, it means none of the supported node types or their values weren't found.
                    # We might not want to create an empty .txt file, or we might. 
                    # For now, we only create a file if a prompt is found.
                    logger.info(f"No specific prompt found in {input_png_file}. No output file created.")
        except json.JSONDecodeError:
            logger.error(f"Could not decode workflow JSON from {input_png_file}.")
    else:
        logger.error(f"Failed to get workflow JSON from {input_png_file}. No output file created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract prompts or full workflow JSON from PNG images with embedded ComfyUI workflow, from a single file or a directory."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", dest="png_file", help="Path to a single PNG image file with embedded workflow.")
    group.add_argument("-d", "--directory", dest="png_directory", help="Path to a directory containing PNG image files.")
    
    parser.add_argument("-o", "--output_dir", help="Directory to save the extracted files. If not specified, output is saved alongside source files.")
    parser.add_argument("-j", "--json", action="store_true", help="Extract full workflow JSON instead of just the prompt text. Creates .json files instead of .txt files.")
    parser.add_argument("--both", action="store_true", help="Write JSON with both positive/negative prompts instead of single text file.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files instead of skipping them. Useful for testing and re-processing.")

    args = parser.parse_args()

    determined_output_dir = None

    if args.output_dir:
        determined_output_dir = os.path.abspath(args.output_dir)
        if not os.path.isdir(determined_output_dir):
            try:
                os.makedirs(determined_output_dir, exist_ok=True)
                logger.info(f"Created output directory: {determined_output_dir}")
            except OSError as e:
                logger.error(f"Could not create specified output directory {determined_output_dir}. {e}")
                exit(1)
    elif args.png_file: # Default for single file: output alongside the file
        determined_output_dir = os.path.dirname(os.path.abspath(args.png_file))
    elif args.png_directory: # Default for directory: output into the source directory
        determined_output_dir = os.path.abspath(args.png_directory)
    
    # This check is crucial to ensure determined_output_dir is valid before proceeding
    if not determined_output_dir or not os.path.isdir(determined_output_dir):
        logger.error(f"Output directory '{determined_output_dir}' is not valid or could not be determined. Please check paths.")
        exit(1)

    if args.png_file:
        if not os.path.isfile(args.png_file):
            logger.error(f"File not found at {args.png_file}")
            exit(1)
        process_png_file(args.png_file, determined_output_dir, args.json, args.both, args.force)
    elif args.png_directory:
        if not os.path.isdir(args.png_directory):
            logger.error(f"Directory not found at {args.png_directory}")
            exit(1)
        
        search_pattern = os.path.join(args.png_directory, '*.png')
        png_files = glob.glob(search_pattern)
        if not png_files:
            logger.info(f"No PNG files found in {args.png_directory} (using pattern: {search_pattern})")
        else:
            logger.info(f"Found {len(png_files)} PNG files in {args.png_directory}.")
            for png_file_path in png_files:
                process_png_file(png_file_path, determined_output_dir, args.json, args.both, args.force)
                logger.info("---") # Separator for multiple files
    
    logger.info("Processing complete.")

# Example of how you might batch process (conceptual):
# import glob
# for png_image_path in glob.glob("path/to/your/images/*.png"):
#     # Call the main logic here, perhaps refactored into a function
#     print(f"Processing {png_image_path}...")
#     # workflow_json_string = get_workflow_from_image(png_image_path)
#     # ... and so on ...
