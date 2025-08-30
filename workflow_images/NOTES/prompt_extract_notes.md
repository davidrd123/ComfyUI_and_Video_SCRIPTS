# Prompt Extract Logic - Final Solution

## The Challenge

ComfyUI workflows use different sampler types that accept text inputs differently:

1. **Standard Samplers** (Basketball JSON): `ClownsharKSampler_Beta`, `KSampler`
   - Inputs: `positive`, `negative` (separate conditioning)
   
2. **WAN Video Samplers** (Graffito JSON): `WanVideoSampler`  
   - Input: `text_embeds` (combined input from bridge)

3. **API vs UI Format**: Different JSON structures require different parsing

## Final Solution: Whitelist + Wiring Verification

### Core Principle
Only process **known sampler types** that are **actually wired** in the workflow.

### Implementation
```python
class UIGraph:
    KNOWN_POSNEG_SAMPLERS = ["KSampler", "ClownsharKSampler_Beta"]
    KNOWN_TEXT_EMBEDS_SAMPLERS = ["WanVideoSampler"]
    
    def sampler_with_posneg(self):
        # Find pos/neg samplers, verify they're wired
        
    def sampler_with_text_embeds(self):
        # Find text_embeds samplers, verify they're wired
```

### Decision Tree (UI Graphs)
1. Try `sampler_with_posneg()` → ClownsharKSampler_Beta, KSampler
2. Try `sampler_with_text_embeds()` → WanVideoSampler  
3. Fallback to single WanVideoTextEncode node

### Key Insights Learned
- **Bridge nodes** like `WanVideoTextEmbedBridge` have pos/neg inputs but aren't samplers
- **Presence ≠ Usage** - nodes can exist but not be wired up
- **Whitelist > Blacklist** - safer to explicitly list known samplers than exclude problematic types

## Implementation Status ✅ COMPLETE

### Core Features Implemented:
- ✅ **Whitelist-based sampler detection** with wiring verification
- ✅ **Three-tier extraction logic**: pos/neg samplers → text_embeds samplers → fallback
- ✅ **Bridge handling**: WanVideoTextEmbedBridge with name-based input tracing
- ✅ **Extensibility hooks**: ENCODER_READERS table for easy new encoder support
- ✅ **Consistent logging**: All print statements replaced with logger calls
- ✅ **Robust PNG handling**: Context management for image files
- ✅ **CLI improvements**: --force flag for testing convenience
- ✅ **Output reachability**: Prefers samplers that reach save/decode nodes
- ✅ **Text normalization**: Consistent whitespace handling across all encoders
- ✅ **Dual prompt output**: --both flag for JSON with positive/negative prompts

### Test Cases:
- **Basketball JSON**: ClownsharKSampler_Beta with separate pos/neg encoders ✅
- **Graffito JSON**: WanVideoSampler with bridge → pos/neg encoders ✅  
- **Levetate JSON**: API format with WanVideoTextEncode ✅

### Extensibility:
- Add new encoders to `ENCODER_READERS` table
- Add new samplers to whitelist constants
- Bridge patterns automatically handled via name-based input tracing
- Add new output node types to `OUTPUT_TYPES` set

### CLI Usage:
```bash
# Extract single prompt (default)
extp_v2 -f image.png

# Extract both positive and negative prompts as JSON
extp_v2 -f image.png --both

# Extract full workflow JSON
extp_v2 -f image.png -j

# Overwrite existing files
extp_v2 -f image.png --force

# Process directory
extp_v2 -d /path/to/images/ --both
```

## Detailed Implementation Changes

### 1. **Added Missing `text_embeds` Branch** 
**Problem**: The `sampler_with_text_embeds()` method was defined but never called in the main extraction logic.

**Solution**: Added complete branch in `extract_prompts()`:
```python
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
```

**Key Features**:
- **Bridge handling**: Traces `WanVideoTextEmbedBridge` inputs by name, not slot numbers
- **Direct WTE support**: Handles cases where `WanVideoTextEncode` feeds directly to sampler
- **Robust error handling**: Checks for wiring before attempting to trace

### 2. **Added Extensibility Hooks**
**Problem**: Adding new encoder types required modifying multiple places in the code.

**Solution**: Created centralized `ENCODER_READERS` table:
```python
# Extensibility: Centralized encoder text readers
ENCODER_READERS = {
    "CLIPTextEncode": lambda n, w, which: _safe_get(w, 0),
    "TextEncodeQwenImageEdit": lambda n, w, which: (_safe_get(w, 0) or "").strip() or None,
    "WanVideoTextEncode": lambda n, w, which: _safe_get(w, 1) if which == "negative" else _safe_get(w, 0),
    # Extension point: Add new encoders here
    # "SomeFutureEncoder": lambda n, w, which: _safe_get(w, 0),
}
```

**Benefits**:
- **Single point of extension**: Add new encoders in one place
- **Consistent interface**: All readers use same signature `(node, widgets, which)`
- **Lambda-based**: Lightweight, no need for separate functions

### 3. **Refactored `read_text_from_node()` Method**
**Before**: Hardcoded if/elif chains for each encoder type
```python
# CLIPTextEncode (UI): text lives at widgets_values[0]
if node_type == "CLIPTextEncode":
    return _safe_get(widgets, 0)

# TextEncodeQwenImageEdit (UI): single prompt at widgets_values[0]
if node_type == "TextEncodeQwenImageEdit":
    text = _safe_get(widgets, 0)
    return text if text and text.strip() else None
```

**After**: Table-driven approach
```python
# Use centralized encoder readers table
reader = self.ENCODER_READERS.get(node_type)
if reader:
    return reader(n, widgets, which)

# Extension point: Add new encoder types to ENCODER_READERS table
logger.debug(f"No reader found for encoder type: {node_type}")
return None
```

### 4. **Consistent Logging System**
**Problem**: Mixed usage of `print()` and `logger` calls throughout the code.

**Solution**: Replaced all `print()` statements with appropriate logger calls:
```python
# Before
print(f"Processing {input_png_file}...")
print(f"Successfully created workflow JSON file {output_file}")
print(f"Warning: Could not set timestamp for {output_file}")

# After  
logger.info(f"Processing {input_png_file}...")
logger.info(f"Successfully created workflow JSON file {output_file}")
logger.warning(f"Could not set timestamp for {output_file}")
```

**Benefits**:
- **Consistent output**: All messages use same format and level
- **Configurable**: Can adjust log level globally
- **Structured**: Includes timestamps and log levels

### 5. **Improved PNG Handling**
**Problem**: Image files weren't properly closed, potential resource leaks.

**Solution**: Added context management:
```python
# Before
img = Image.open(image_path)
if img.info and isinstance(img.info, dict):
    # ... process image
return None

# After
with Image.open(image_path) as img:
    if img.info and isinstance(img.info, dict):
        # ... process image
    return None
```

**Benefits**:
- **Resource safety**: Images automatically closed
- **Exception safety**: Proper cleanup even on errors
- **Best practice**: Follows Python context manager patterns

### 6. **Enhanced CLI with --force Flag**
**Problem**: No way to overwrite existing files during testing.

**Solution**: Added `--force` flag:
```python
parser.add_argument("--force", action="store_true", 
                   help="Overwrite existing output files instead of skipping them. Useful for testing and re-processing.")
```

**Usage**:
```bash
# Skip existing files (default)
extp_v2 -f image.png

# Overwrite existing files
extp_v2 -f image.png --force
```

### 7. **Improved Error Handling**
**Problem**: Generic error messages, hard to debug issues.

**Solution**: Enhanced error handling with specific messages:
```python
# Before
except OSError as e:
    print(f"Warning: Error setting timestamp for {output_file}: {e}")

# After
except OSError as e:
    if e.errno == 1: # Operation not permitted
        logger.warning(f"Could not set timestamp for {output_file} (Operation not permitted). File created with current timestamp.")
    else:
        logger.warning(f"Error setting timestamp for {output_file}: {e}")
```

### 8. **Text Normalization**
**Problem**: Inconsistent whitespace handling across different encoders.

**Solution**: Added `_norm()` function and applied to all encoders:
```python
def _norm(s):
    """Normalize text: strip whitespace, return None if empty."""
    return (s or "").strip() or None

ENCODER_READERS = {
    "CLIPTextEncode": lambda n, w, which: _norm(_safe_get(w, 0)),
    "TextEncodeQwenImageEdit": lambda n, w, which: _norm(_safe_get(w, 0)),
    "WanVideoTextEncode": lambda n, w, which: _norm(_safe_get(w, 1) if which == "negative" else _safe_get(w, 0)),
}
```

**Benefits**:
- **Consistent output**: No trailing spaces or blank lines
- **Clean data**: Normalized text for downstream processing
- **Robust parsing**: Handles edge cases gracefully

### 9. **Output Reachability Detection**
**Problem**: Multiple samplers in one graph - which one to choose?

**Solution**: Added `_reaches_output()` method to prefer samplers that feed save/decode nodes:
```python
def _reaches_output(self, start_id, max_hops=200):
    """Check if a node has a path to an output node (save/decode)."""
    # Build outgoing links index: src_id -> [dst_ids]
    out_by_src = {}
    for L in self.links:
        out_by_src.setdefault(L[1], []).append(L[3])
    
    # BFS from start_id to find OUTPUT_TYPES nodes
    seen, q = set([start_id]), [start_id]
    hops = 0
    while q and hops < max_hops:
        nxt = []
        for nid in q:
            for dst in out_by_src.get(nid, []):
                if dst in seen: continue
                if (self.nodes.get(dst) or {}).get("type") in self.OUTPUT_TYPES:
                    return True
                seen.add(dst); nxt.append(dst)
        q = nxt; hops += 1
    return False
```

**Benefits**:
- **Smart selection**: Chooses the "active" sampler in complex graphs
- **Mirrors workflow**: Follows actual data flow to output
- **Fallback safe**: Uses first sampler if none reach output

### 10. **Dual Prompt Output (--both flag)**
**Problem**: Single prompt output limits automation and analysis.

**Solution**: Added `--both` flag for structured JSON output:
```python
# New flag
parser.add_argument("--both", action="store_true", 
                   help="Write JSON with both positive/negative prompts.")

# Usage
extp_v2 -f image.png --both
# Creates: image.json with {"positive": "...", "negative": "..."}
```

**Benefits**:
- **Structured data**: JSON format for programmatic access
- **Complete information**: Both prompts preserved
- **Idempotent**: Consistent output format for automation

## Architecture Overview

### Core Classes and Methods

#### `UIGraph` Class
**Purpose**: Normalizes ComfyUI workflow graphs for robust querying

**Key Methods**:
- `__init__(wf)`: Builds fast lookup tables for nodes and links
- `sampler_with_posneg()`: Finds samplers with positive/negative inputs
- `sampler_with_text_embeds()`: Finds samplers with text_embeds input
- `input_index(node, name)`: Maps input names to slot indices
- `src_for(dst_id, dst_slot)`: Finds source node for an input slot
- `read_text_from_node(nid, which)`: Reads text from encoder nodes

#### `extract_prompts()` Function
**Purpose**: Main extraction logic with three-tier decision tree

**Decision Tree**:
1. **UI Graph Path** (has "nodes" array):
   - (a) Sampler with pos/neg inputs → trace to encoders
   - (b) Sampler with text_embeds → trace to bridge or WTE
   - (c) Fallback to standalone WanVideoTextEncode
2. **API Graph Path** (flat dict):
   - Direct access to `inputs.positive_prompt/negative_prompt`

### Data Flow
```
PNG Image → Extract JSON → Parse Workflow → UIGraph → Extract Prompts → Save File
```

### Extension Points
1. **New Encoders**: Add to `ENCODER_READERS` table
2. **New Samplers**: Add to `KNOWN_POSNEG_SAMPLERS` or `KNOWN_TEXT_EMBEDS_SAMPLERS`
3. **New Bridge Patterns**: Automatically handled via name-based input tracing

## Testing Strategy

### Manual Testing
- **Basketball JSON**: Verify ClownsharKSampler_Beta extraction
- **Graffito JSON**: Verify WanVideoSampler + bridge extraction  
- **Levetate JSON**: Verify API format extraction

### Automated Testing (Future)
- Unit tests for each encoder reader
- Integration tests for each workflow type
- Edge case testing (unwired nodes, malformed JSON)

## Performance Considerations

### Optimizations Implemented
- **O(1) node lookup**: Pre-built `self.nodes` dictionary
- **O(1) link lookup**: Pre-built `self.in_edge` reverse index
- **Early returns**: Stop processing once prompts found
- **Context management**: Proper resource cleanup

### Scalability
- **Memory efficient**: No deep copying of workflow data
- **CPU efficient**: Single pass through nodes/links during init
- **Extensible**: Table-driven approach scales with new types
