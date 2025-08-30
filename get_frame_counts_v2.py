import cv2
import glob
import os
import argparse
import subprocess
import shutil
import json

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def get_frame_count_ffprobe(video_path):
    """
    Gets the frame count of a video file using ffprobe.

    Args:
        video_path (str): The path to the video file.

    Returns:
        int: The frame count, or None if ffprobe fails or the count cannot be determined.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-of', 'json',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        output = json.loads(result.stdout)
        # Check if streams array exists and has elements
        if output and 'streams' in output and len(output['streams']) > 0:
            # Check if nb_read_frames exists, otherwise try nb_frames as a fallback
            if 'nb_read_frames' in output['streams'][0]:
                frame_count_str = output['streams'][0]['nb_read_frames']
                if frame_count_str != 'N/A':
                    return int(frame_count_str)
            elif 'nb_frames' in output['streams'][0]: # Fallback for some containers
                frame_count_str = output['streams'][0]['nb_frames']
                if frame_count_str != 'N/A':
                     print(f"  Note: Using fallback 'nb_frames' for {os.path.basename(video_path)}")
                     return int(frame_count_str)

        print(f"Warning: Could not extract frame count using ffprobe for {os.path.basename(video_path)}. Output: {result.stdout.strip()}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe for {os.path.basename(video_path)}: {e}")
        print(f"ffprobe stderr: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
         # This specific check happens earlier now, but keep for safety
        print("Error: ffprobe command not found. Make sure FFmpeg is installed and in your PATH.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred with ffprobe for {os.path.basename(video_path)}: {e}")
        return None

def get_video_frame_counts(directory=".", use_ffprobe=False):
    """
    Finds all mp4 files in the specified directory, reads their frame counts
    using cv2 and optionally ffprobe, and returns a sorted list of unique frame counts (from cv2).

    Args:
        directory (str): The directory to search for mp4 files. Defaults to CWD.
        use_ffprobe (bool): Whether to also use ffprobe to check frame counts.

    Returns:
        list: A sorted list of unique frame counts (from cv2) found, or None if cv2 is not available.
              Returns an empty list if no mp4 files are found or none can be read by cv2.
    """
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is required but not installed.")
        print("Please install it using: pip install opencv-python")
        return None

    # Check for ffprobe availability if requested
    ffprobe_available = use_ffprobe and is_tool('ffprobe')
    if use_ffprobe and not ffprobe_available:
        print("Warning: ffprobe requested but not found in PATH. Frame counts will not be double-checked.")
        use_ffprobe = False # Disable ffprobe use if not found


    # Use absolute path for clarity in messages and glob
    abs_directory = os.path.abspath(directory)
    # Ensure we only grab mp4 files
    search_pattern = os.path.join(abs_directory, "*.mp4")
    mp4_files = glob.glob(search_pattern)
    # Filter out potential directories named *.mp4
    mp4_files = [f for f in mp4_files if os.path.isfile(f)]

    frame_counts_cv2 = [] # Store counts obtained by cv2

    if not mp4_files:
        print(f"No .mp4 files found in {abs_directory}")
        return []

    print(f"Found {len(mp4_files)} mp4 files in '{abs_directory}'. Checking frame counts...")

    discrepancies = 0

    for video_path in mp4_files:
        # Use relative path for display if it was originally relative
        display_path = os.path.relpath(video_path, os.getcwd())

        # --- Get frame count using OpenCV (cv2) ---
        cap = cv2.VideoCapture(video_path)
        frame_count_cv2 = -1 # Default to -1 if not opened or count fails
        if not cap.isOpened():
            print(f"Warning: [cv2] Could not open video file: {display_path}")
        else:
            frame_count_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count_cv2 <= 0:
                print(f"Warning: [cv2] Could not get frame count for: {display_path} (Result: {frame_count_cv2})")
            else:
                frame_counts_cv2.append(frame_count_cv2) # Add to list only if valid
            cap.release() # Release capture object

        # --- Get frame count using ffprobe (if requested and available) ---
        frame_count_ffprobe = None
        if use_ffprobe and ffprobe_available:
             frame_count_ffprobe = get_frame_count_ffprobe(video_path)

        # --- Print results ---
        if frame_count_cv2 > 0:
            if use_ffprobe and ffprobe_available:
                ffprobe_str = f"{frame_count_ffprobe}" if frame_count_ffprobe is not None else "N/A"
                print(f"  {display_path}: cv2={frame_count_cv2}, ffprobe={ffprobe_str}", end="")
                if frame_count_ffprobe is not None and frame_count_cv2 != frame_count_ffprobe:
                    print("  <- DISCREPANCY")
                    discrepancies += 1
                else:
                    print() # Newline if no discrepancy
            else:
                # Print only cv2 result if ffprobe not used/available
                print(f"  {display_path}: {frame_count_cv2} frames (cv2)")
        # else: # Warnings for cv2 failures are printed above

        # Print ffprobe warning only if cv2 also failed or if ffprobe failed when cv2 succeeded
        if use_ffprobe and ffprobe_available and frame_count_ffprobe is None and frame_count_cv2 <= 0:
             # Avoid double warning if ffprobe func already printed one
             pass # get_frame_count_ffprobe handles its own warnings
        elif use_ffprobe and ffprobe_available and frame_count_ffprobe is None and frame_count_cv2 > 0:
             print(f"  Warning: [ffprobe] failed for {display_path} (cv2 count: {frame_count_cv2})")


    if not frame_counts_cv2:
        print("Could not retrieve valid frame counts from any video files using cv2.")
        return []

    if use_ffprobe and discrepancies > 0:
        print(f"\nWarning: Found {discrepancies} discrepancies between cv2 and ffprobe frame counts.")

    unique_frame_counts = sorted(list(set(frame_counts_cv2)))
    return unique_frame_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get unique frame counts from MP4 videos in a directory, optionally using ffprobe to double-check.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="The directory containing the mp4 files. Defaults to the current working directory.",
    )
    # Added argument for ffprobe
    parser.add_argument(
        "--ffprobe",
        action="store_true",
        help="Use ffprobe (if available in PATH) to double-check frame counts.",
    )
    args = parser.parse_args()

    # Pass ffprobe flag to the function
    unique_counts = get_video_frame_counts(args.directory, use_ffprobe=args.ffprobe)

    if unique_counts is not None:
        if unique_counts:
            print("\nUnique frame counts found (using cv2):")
            print(unique_counts)
        # No need for else here, messages are handled within the function 