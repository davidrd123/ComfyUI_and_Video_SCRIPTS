#!/bin/bash

# --- Configuration ---
SRC_DIR="/mnt/i/VIDEO/1080p"
# Set your desired target width and height
TARGET_W=784
TARGET_H=432
# Set a destination directory name reflecting the target size
DST_DIR="/mnt/i/VIDEO/1080p/CONVERTED/${TARGET_W}x${TARGET_H}"

# --- Script Logic ---
# Ensure destination directory exists
mkdir -p "$DST_DIR"
if [ ! -d "$DST_DIR" ]; then
    echo "Error: Could not create destination directory: $DST_DIR"
    exit 1
fi

# Use print0/read -d for robust filename handling
find "$SRC_DIR" -maxdepth 1 -type f -iname '*.mov' -print0 | while IFS= read -r -d $'\0' F; do
  # Check if file exists (paranoia check for find/read issues)
  if [ ! -f "$F" ]; then
    echo "Warning: Skipping non-existent file listed by find: $F"
    continue
  fi

  base=$(basename "$F" .mov)
  output_file="$DST_DIR/${base}_${TARGET_W}x${TARGET_H}.mp4"
  echo "Processing $F -> ${output_file}"

  # Pure CPU encode using libx265 for HEVC
  ffmpeg -y \
    -i "$F" \
    -map 0:v:0 -map 0:a? -map 0:s? \
    -vf "scale=w=$TARGET_W:h=$TARGET_H:flags=lanczos,format=pix_fmts=yuv420p10le" \
    -c:v libx265 \
        -preset medium \
        -crf 20 \
        -x265-params "aq-mode=1" \
    -c:a copy -c:s copy \
    -tag:v hvc1 \
    -movflags +faststart \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
    "$output_file"

  # Check ffmpeg exit status
  if [ $? -ne 0 ]; then
      echo "Error processing $F"
      # Optional: stop script on error
      # exit 1
  fi

done

echo "Conversion complete. Files saved to $DST_DIR"

