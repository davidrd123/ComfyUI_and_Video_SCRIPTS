import cv2
import glob
import os

def get_video_frame_counts(directory="."):
    """
    Finds all mp4 files in the specified directory, reads their frame counts,
    and returns a sorted list of unique frame counts.

    Args:
        directory (str): The directory to search for mp4 files. Defaults to CWD.

    Returns:
        list: A sorted list of unique frame counts found, or None if cv2 is not available.
              Returns an empty list if no mp4 files are found or none can be read.
    """
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python is required but not installed.")
        print("Please install it using: pip install opencv-python")
        return None

    mp4_files = glob.glob(os.path.join(directory, "*.mp4"))
    frame_counts = []

    if not mp4_files:
        print(f"No .mp4 files found in {os.path.abspath(directory)}")
        return []

    print(f"Found {len(mp4_files)} mp4 files. Checking frame counts...")

    for video_path in mp4_files:
        # Use absolute path for clarity in messages
        abs_video_path = os.path.abspath(video_path)
        cap = cv2.VideoCapture(abs_video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {abs_video_path}")
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            frame_counts.append(frame_count)
            print(f"  {video_path}: {frame_count} frames")
        else:
             print(f"Warning: Could not get frame count for: {abs_video_path} (Result: {frame_count})")
        cap.release()

    if not frame_counts:
        print("Could not retrieve frame counts from any video files.")
        return []

    unique_frame_counts = sorted(list(set(frame_counts)))
    return unique_frame_counts

if __name__ == "__main__":
    # Use the directory of the script itself if run directly, or CWD otherwise
    script_dir = os.path.dirname(__file__) if "__file__" in locals() else "."
    unique_counts = get_video_frame_counts(script_dir)

    if unique_counts is not None:
        if unique_counts:
            print("\nUnique frame counts found:")
            print(unique_counts)
        # No need for else here, messages are handled within the function
