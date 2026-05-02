"""
MediaPipe Hands Analysis Script
Measures runtime performance, inference time, and tracking consistency.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration
VIDEO_SOURCE = "videos/BenchmarkVideo.mp4"  # Video file path
OUTPUT_DIR = "MediaPipeHands/analysis_output"

# Analysis metrics
class MetricsTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        
        # Timing metrics
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.total_start = None
        
        # Tracking consistency metrics
        self.total_frames = 0
        self.total_hands_detected = 0
        self.expected_hands = 0  # 2 hands per frame
        self.frames_with_hands = 0
        self.frames_without_hands = 0
        
        # Hand count distribution
        self.single_hand_count = 0
        self.multi_hand_count = 0
        
    def update_timing(self, inference_time, frame_time):
        self.inference_times.append(inference_time)
        self.frame_times.append(frame_time)
        
    def update_tracking(self, num_hands, landmarks_per_hand):
        self.total_frames += 1
        self.expected_hands += 2  # Expected 2 hands per frame
        self.total_hands_detected += num_hands
        
        if num_hands == 0:
            self.frames_without_hands += 1
        else:
            self.frames_with_hands += 1
            
            if num_hands == 1:
                self.single_hand_count += 1
            else:
                self.multi_hand_count += 1
                

                        
    def get_summary(self):
        avg_inference = np.mean(self.inference_times) * 1000 if self.inference_times else 0
        avg_frame = np.mean(self.frame_times) * 1000 if self.frame_times else 0
        avg_fps = 1.0 / np.mean(self.frame_times) if self.frame_times and np.mean(self.frame_times) > 0 else 0
        
        # Tracking consistency
        tracking_rate = (self.total_hands_detected / self.expected_hands * 100) if self.expected_hands > 0 else 0
        
        return {
            "avg_inference_ms": avg_inference,
            "avg_frame_ms": avg_frame,
            "avg_fps": avg_fps,
            "total_frames": self.total_frames,
            "total_hands_detected": self.total_hands_detected,
            "expected_hands": self.expected_hands,
            "frames_with_hands": self.frames_with_hands,
            "frames_without_hands": self.frames_without_hands,
            "tracking_rate_percent": tracking_rate,
            "single_hand_count": self.single_hand_count,
            "multi_hand_count": self.multi_hand_count
        }
        
    def print_live_stats(self, metrics):
        """Print live statistics to console."""
        print(f"\rFPS: {metrics['avg_fps']:.1f} | "
              f"Inf: {metrics['avg_inference_ms']:.1f}ms | "
              f"Track: {metrics['tracking_rate_percent']:.1f}%", 
              end="", flush=True)


def print_detailed_report(metrics):
    """Print a detailed analysis report."""
    print("\n" + "="*60)
    print("MEDIAPIPE HANDS - DETAILED ANALYSIS REPORT")
    print("="*60)
    
    print("\n📊 TIMING METRICS")
    print("-"*40)
    print(f"  Average Inference Time:  {metrics['avg_inference_ms']:.2f} ms")
    print(f"  Average Frame Time:      {metrics['avg_frame_ms']:.2f} ms")
    print(f"  Average FPS:             {metrics['avg_fps']:.2f}")
    
    print("\n🔍 TRACKING CONSISTENCY")
    print("-"*40)
    print(f"  Total Frames Processed:  {metrics['total_frames']}")
    print(f"  Hands Detected:          {metrics['total_hands_detected']}")
    print(f"  Expected Hands:         {metrics['expected_hands']} (2 per frame)")
    print(f"  Tracking Rate:           {metrics['tracking_rate_percent']:.2f}%")
    print(f"  Frames with Hands:       {metrics['frames_with_hands']}")
    print(f"  Frames without Hands:    {metrics['frames_without_hands']}")
    
    print("\n👋 HAND COUNT DISTRIBUTION")
    print("-"*40)
    print(f"  Single Hand Frames:      {metrics['single_hand_count']}")
    print(f"  Multi-Hand Frames:       {metrics['multi_hand_count']}")
    
    print("\n" + "="*60)


def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize metrics tracker
    metrics = MetricsTracker(window_size=30)
    metrics.total_start = time.time()
    
    # Open video file
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video source: {VIDEO_SOURCE} ({width}x{height} @ {fps}fps)")
    print("Press 'Q' to quit, 'Space' to pause/resume, 'S' to print summary")
    
    # Initialize MediaPipe Hands
    paused = False
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            frame_start = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret:
                print("\nError: Failed to read frame")
                break
            
            # Convert BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inference
            inf_start = time.perf_counter()
            results = hands.process(rgb)
            inf_end = time.perf_counter()
            inference_time = inf_end - inf_start
            
            # Analyze results
            num_hands = 0
            landmarks_per_hand = []
            
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks_per_hand.append(hand_landmarks)
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
            
            # Update metrics
            frame_end = time.perf_counter()
            frame_time = frame_end - frame_start
            
            metrics.update_timing(inference_time, frame_time)
            metrics.update_tracking(num_hands, landmarks_per_hand)
            
            # Display live stats on frame
            live_metrics = metrics.get_summary()
            stats_text = f"FPS: {live_metrics['avg_fps']:.1f} | Inf: {live_metrics['avg_inference_ms']:.1f}ms | Track: {live_metrics['tracking_rate_percent']:.1f}%"
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("MediaPipe Hands - Analysis Mode", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                print_detailed_report(live_metrics)
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
            
            if paused:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):
                    paused = False
                    print("Resumed")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final report
    final_metrics = metrics.get_summary()
    print_detailed_report(final_metrics)
    
    # Save report to file
    report_path = os.path.join(OUTPUT_DIR, "hand_tracking_report.txt")
    with open(report_path, 'w') as f:
        f.write("MEDIAPIPE HANDS ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Runtime: {time.time() - metrics.total_start:.2f}s\n")
        f.write(f"Total Frames: {final_metrics['total_frames']}\n")
        f.write(f"Hands Detected: {final_metrics['total_hands_detected']}\n")
        f.write(f"Expected Hands: {final_metrics['expected_hands']}\n")
        f.write(f"Average FPS: {final_metrics['avg_fps']:.2f}\n")
        f.write(f"Average Inference Time: {final_metrics['avg_inference_ms']:.2f}ms\n")
        f.write(f"Tracking Rate: {final_metrics['tracking_rate_percent']:.2f}%\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()