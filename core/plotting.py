# core/plotting.py
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# --- ADD LOGGING IMPORTS ---
from .logging import log_info, log_warning, log_error
from .constants import EMO_VAL # Use the centralized EMO_VAL

EmotionTimelinePoint = Dict[str, Any]
SpeakerStats = Dict[str, Any]
EmotionSummary = Dict[str, SpeakerStats]

EMOTION_COLORS: Dict[str, str] = { # ... (colors) ...
    'joy': 'gold', 'neutral': 'gray', 'sadness': 'blue', 'anger': 'red',
    'surprise': 'orange', 'fear': 'purple', 'disgust': 'brown', 'love': 'pink',
    'unknown': 'black', 'analysis_skipped': 'lightgrey',
    'analysis_failed': 'darkred', 'no_text': 'whitesmoke'
}

def _save_plot(figure: Figure, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
        log_info(f"Saved plot: {output_path}") # USE LOG_INFO
    except Exception as e:
        log_error(f"Failed to save plot {output_path}: {e}") # USE LOG_ERROR
    finally:
        plt.close(figure)

def plot_emotion_trajectory(summary_data: EmotionSummary, output_dir: Path, file_prefix: str) -> List[Path]:
    plot_files: List[Path] = []
    for speaker, stats in summary_data.items():
        timeline: List[EmotionTimelinePoint] = stats.get("emotion_timeline", [])
        if not timeline or len(timeline) < 2:
            log_warning(f"Skipping trajectory plot for speaker '{speaker}': insufficient timeline data.") # USE LOG_WARNING
            continue
        # ... (plotting logic using EMO_VAL) ...
        times: List[float] = [point.get("time", float(i)) for i, point in enumerate(timeline)]
        emotions: List[str] = [point.get("emotion", "unknown") for point in timeline]
        fig: Figure; ax: Axes
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times, emotions, marker="o", linestyle="-", markersize=4)
        unique_emotions = sorted(list(set(emotions)), key=lambda e: EMO_VAL.get(e, 0.0))
        ax.set_yticks(unique_emotions)
        ax.set_title(f"Emotion Trajectory for {speaker}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Emotion")
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        filename: str = f"{file_prefix}_{speaker}_emotion_trajectory.png"
        output_path: Path = output_dir / filename
        _save_plot(fig, output_path)
        plot_files.append(output_path)
    return plot_files

def plot_emotion_distribution(summary_data: EmotionSummary, output_dir: Path, file_prefix: str) -> List[Path]:
    plot_files: List[Path] = []
    for speaker, stats in summary_data.items():
        timeline: List[EmotionTimelinePoint] = stats.get("emotion_timeline", [])
        if not timeline:
             log_warning(f"Skipping distribution plot for speaker '{speaker}': missing timeline data in summary.") # USE LOG_WARNING
             continue
        # ... (plotting logic) ...
        emos: List[str] = [point.get("emotion", "unknown") for point in timeline]
        counts: Counter = Counter(emos)
        if not counts: continue
        labels: Tuple[str, ...]; values: Tuple[int, ...]
        sorted_items = sorted(counts.items())
        if not sorted_items: continue
        labels, values = zip(*sorted_items)
        colors: List[str] = [EMOTION_COLORS.get(label, "black") for label in labels]
        fig: Figure; ax: Axes
        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, pctdistance=0.85
        )
        ax.set_title(f"Emotion Distribution for {speaker}")
        plt.tight_layout()
        filename: str = f"{file_prefix}_{speaker}_emotion_distribution.png"
        output_path: Path = output_dir / filename
        _save_plot(fig, output_path)
        plot_files.append(output_path)
    return plot_files

def plot_emotion_volatility(summary_data: EmotionSummary, output_dir: Path, file_prefix: str) -> List[Path]:
    speakers: List[str] = list(summary_data.keys())
    volatility: List[float] = [float(summary_data[spk].get("emotion_volatility", 0.0)) for spk in speakers]
    if not speakers or not any(v > 0 for v in volatility):
        log_warning("Skipping volatility plot: No speakers or no volatility data > 0.") # USE LOG_WARNING
        return []
    # ... (plotting logic) ...
    sorted_pairs = sorted(zip(speakers, volatility), key=lambda item: item[1], reverse=True)
    sorted_speakers, sorted_volatility = zip(*sorted_pairs) if sorted_pairs else ([], [])
    fig: Figure; ax: Axes
    fig, ax = plt.subplots(figsize=(max(6, len(sorted_speakers) * 0.8), 6))
    ax.bar(sorted_speakers, sorted_volatility, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Emotion Volatility (StdDev of Scores)")
    ax.set_title("Emotion Volatility by Speaker")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename: str = f"{file_prefix}_emotion_volatility.png"
    output_path: Path = output_dir / filename
    _save_plot(fig, output_path)
    return [output_path]

def plot_emotion_score_timeline(summary_data: EmotionSummary, output_dir: Path, file_prefix: str) -> List[Path]:
    plot_files: List[Path] = []
    for speaker, stats in summary_data.items():
        timeline: List[EmotionTimelinePoint] = stats.get("emotion_timeline", [])
        if not timeline or len(timeline) < 2:
            log_warning(f"Skipping intensity score plot for speaker '{speaker}': insufficient timeline data.") # USE LOG_WARNING
            continue
        # ... (plotting logic using EMO_VAL) ...
        times: List[float] = [point.get("time", float(i)) for i, point in enumerate(timeline)]
        scores: List[float] = [EMO_VAL.get(point.get("emotion", "unknown"), 0.0) for point in timeline]
        fig: Figure; ax: Axes
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times, scores, marker="x", linestyle="--", color="darkgreen", markersize=5)
        ax.set_title(f"Emotion Intensity Score Timeline for {speaker}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Emotion Intensity Score")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='grey', linewidth=0.8)
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()
        filename: str = f"{file_prefix}_{speaker}_emotion_intensity.png"
        output_path: Path = output_dir / filename
        _save_plot(fig, output_path)
        plot_files.append(output_path)
    return plot_files

def generate_all_plots(summary_data: EmotionSummary, output_dir_str: str, job_id_suffix: str) -> List[str]:
    output_dir = Path(output_dir_str)
    file_prefix: str = f"plots_{job_id_suffix}"
    all_plot_files_paths: List[Path] = []

    log_info(f"Generating plots with prefix '{file_prefix}' in directory: {output_dir}") # USE LOG_INFO

    if not summary_data:
        log_warning("No summary data provided to generate_all_plots. Skipping plot generation.") # USE LOG_WARNING
        return []

    plot_functions = [ # ... (plot functions list) ...
        plot_emotion_trajectory, plot_emotion_distribution,
        plot_emotion_volatility, plot_emotion_score_timeline,
    ]

    for plot_func in plot_functions:
        try:
            plot_paths: List[Path] = plot_func(summary_data, output_dir, file_prefix)
            all_plot_files_paths.extend(plot_paths)
        except Exception as e:
            log_error(f"Failed during {plot_func.__name__} generation: {e}\n{traceback.format_exc()}") # USE LOG_ERROR

    log_info(f"Plot generation finished. Generated {len(all_plot_files_paths)} plot file(s).") # USE LOG_INFO
    return [str(p) for p in all_plot_files_paths]