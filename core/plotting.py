# core/plotting.py
import os
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting (prevents errors in environments without display)
import matplotlib.pyplot as plt
from collections import Counter
import traceback # For detailed error logging

# Import logging functions if you want module-level logging
# from .logging import log_info, log_error

# Define constants
EMOTION_COLORS = {
    'joy': 'gold',
    'neutral': 'gray',
    'sadness': 'blue',
    'anger': 'red',
    'surprise': 'orange',
    'fear': 'purple',
    'disgust': 'brown', # Add if applicable
    'love': 'pink',   # Add if applicable
    'unknown': 'black',
    'analysis_skipped': 'lightgrey',
    'analysis_failed': 'darkred',
    'no_text': 'whitesmoke'
}

# Define emotion values for intensity plot (consistent with pipeline calculation)
EMO_VAL = {'joy': 1, 'neutral': 0, 'sadness': -1, 'anger': -2, 'surprise': 0.5, 'fear': -1.5}

def _save_plot(figure, output_path):
    """Helper function to save and close a matplotlib figure."""
    try:
        figure.savefig(output_path)
        # print(f"INFO: Saved plot: {output_path}") # Optional logging
    except Exception as e:
        print(f"ERROR: Failed to save plot {output_path}: {e}")
        # Log error appropriately
    finally:
        plt.close(figure) # Ensure figure is closed to free memory

def plot_emotion_trajectory(summary_data, output_dir, file_prefix):
    """Generates and saves emotion trajectory plots for each speaker."""
    plot_files = []
    for speaker, stats in summary_data.items():
        timeline = stats.get("emotion_timeline", [])
        if not timeline or len(timeline) < 2: # Need at least 2 points to plot a line
            print(f"WARN: Skipping trajectory plot for speaker '{speaker}': insufficient timeline data.")
            continue

        times = [point.get("time", i) for i, point in enumerate(timeline)] # Use index if time is missing
        emotions = [point.get("emotion", "unknown") for point in timeline]

        fig, ax = plt.subplots(figsize=(12, 5)) # Create figure and axes
        ax.plot(times, emotions, marker="o", linestyle="-", markersize=4)
        
        # Use a defined list or dynamically get unique emotions for y-ticks
        unique_emotions = sorted(list(set(emotions)), key=lambda e: EMO_VAL.get(e, 0))
        ax.set_yticks(unique_emotions) # Set y-ticks to actual emotions present

        ax.set_title(f"Emotion Trajectory for {speaker}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Emotion")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7) # Grid lines for y-axis
        plt.xticks(rotation=30) # Rotate x-axis labels if needed
        plt.tight_layout() # Adjust layout

        filename = f"{file_prefix}_{speaker}_emotion_trajectory.png"
        output_path = os.path.join(output_dir, filename)
        _save_plot(fig, output_path)
        plot_files.append(output_path)
    return plot_files

def plot_emotion_distribution(summary_data, output_dir, file_prefix):
    """Generates and saves emotion distribution pie charts for each speaker."""
    plot_files = []
    for speaker, stats in summary_data.items():
        # Use the full timeline data if available, otherwise need to derive from segments
        timeline = stats.get("emotion_timeline", [])
        if not timeline:
             # Fallback: If timeline wasn't stored, could potentially recalculate from segments if passed
             print(f"WARN: Skipping distribution plot for speaker '{speaker}': missing timeline data in summary.")
             continue

        emos = [point.get("emotion", "unknown") for point in timeline]
        counts = Counter(emos)
        if not counts: # Skip if no emotions counted
            continue

        labels, values = zip(*sorted(counts.items())) # Sort for consistent order
        colors = [EMOTION_COLORS.get(label, "black") for label in labels]

        fig, ax = plt.subplots(figsize=(8, 8)) # Create figure and axes
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, pctdistance=0.85)
        ax.set_title(f"Emotion Distribution for {speaker}")
        plt.tight_layout() # Adjust layout

        filename = f"{file_prefix}_{speaker}_emotion_distribution.png"
        output_path = os.path.join(output_dir, filename)
        _save_plot(fig, output_path)
        plot_files.append(output_path)
    return plot_files

def plot_emotion_volatility(summary_data, output_dir, file_prefix):
    """Generates and saves a bar chart comparing emotion volatility across speakers."""
    speakers = list(summary_data.keys())
    volatility = [summary_data[spk].get("emotion_volatility", 0) for spk in speakers]

    if not speakers or not any(v > 0 for v in volatility): # Check if data exists
        print("WARN: Skipping volatility plot: No speakers or no volatility data.")
        return []

    fig, ax = plt.subplots(figsize=(max(6, len(speakers) * 0.8), 6)) # Adjust width based on speaker count
    ax.bar(speakers, volatility, color="skyblue")
    plt.xticks(rotation=45, ha="right") # Rotate labels for readability
    ax.set_ylabel("Emotion Volatility (StdDev of Scores)")
    ax.set_title("Emotion Volatility by Speaker")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7) # Add y-axis grid
    plt.tight_layout() # Adjust layout

    filename = f"{file_prefix}_emotion_volatility.png"
    output_path = os.path.join(output_dir, filename)
    _save_plot(fig, output_path)
    return [output_path] # Returns list with one path

def plot_emotion_score_timeline(summary_data, output_dir, file_prefix):
    """Generates and saves emotion intensity score timelines for each speaker."""
    plot_files = []
    for speaker, stats in summary_data.items():
        timeline = stats.get("emotion_timeline", [])
        if not timeline or len(timeline) < 2: # Need at least 2 points
            print(f"WARN: Skipping intensity plot for speaker '{speaker}': insufficient timeline data.")
            continue

        times = [point.get("time", i) for i, point in enumerate(timeline)]
        scores = [EMO_VAL.get(point.get("emotion", "unknown"), 0) for point in timeline] # Default unknown to 0

        fig, ax = plt.subplots(figsize=(12, 5)) # Create figure and axes
        ax.plot(times, scores, marker="x", linestyle="--", color="darkgreen", markersize=5)
        ax.set_title(f"Emotion Intensity Score Timeline for {speaker}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Emotion Intensity Score")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='grey', linewidth=0.8) # Add zero line
        plt.xticks(rotation=30) # Rotate x-axis labels if needed
        plt.tight_layout() # Adjust layout

        filename = f"{file_prefix}_{speaker}_emotion_intensity.png"
        output_path = os.path.join(output_dir, filename)
        _save_plot(fig, output_path)
        plot_files.append(output_path)
    return plot_files

def generate_all_plots(summary_data, output_dir, job_id_suffix):
    """
    Generates all defined emotion plots based on summary data.

    Args:
        summary_data (dict): The dictionary containing speaker summaries (including timelines).
        output_dir (str): The directory to save the plot images.
        job_id_suffix (str): A suffix (like job_id or job_id_relabeled) for filenames.

    Returns:
        list: A list of full paths to the generated plot files.
    """
    all_plot_files = []
    file_prefix = f"plots_{job_id_suffix}" # Consistent prefix for plot files

    print(f"INFO: Generating plots with prefix '{file_prefix}' in directory: {output_dir}")

    # Ensure summary_data is not empty
    if not summary_data:
        print("WARN: No summary data provided to generate_all_plots. Skipping plot generation.")
        return all_plot_files

    # --- Call individual plotting functions ---
    try:
        all_plot_files.extend(plot_emotion_trajectory(summary_data, output_dir, file_prefix))
    except Exception as e:
        print(f"ERROR: Failed during trajectory plot generation: {e}\n{traceback.format_exc()}")

    try:
        all_plot_files.extend(plot_emotion_distribution(summary_data, output_dir, file_prefix))
    except Exception as e:
        print(f"ERROR: Failed during distribution plot generation: {e}\n{traceback.format_exc()}")

    try:
        all_plot_files.extend(plot_emotion_volatility(summary_data, output_dir, file_prefix))
    except Exception as e:
        print(f"ERROR: Failed during volatility plot generation: {e}\n{traceback.format_exc()}")

    try:
        all_plot_files.extend(plot_emotion_score_timeline(summary_data, output_dir, file_prefix))
    except Exception as e:
        print(f"ERROR: Failed during score timeline plot generation: {e}\n{traceback.format_exc()}")

    print(f"INFO: Plot generation finished. Generated {len(all_plot_files)} plot file(s).")
    return all_plot_files