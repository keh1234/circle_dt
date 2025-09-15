import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import os

# --- Constants ---
NUM_CAMERAS = 6
MM_PER_PIXEL = 0.1
REAL_CIRCLE_RADIUS_MM = 6.4
FOCAL_LENGTH_MM = 3910.31
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
HISTORY_SIZE = 5
MASTER_EXCEL_FILENAME = "circle_log.xlsx"

# Colors (BGR)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)

# Font settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Hough Circle parameters
HOUGH_DP = 1
HOUGH_MIN_DIST = 50
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 18
HOUGH_MIN_RADIUS = 50
HOUGH_MAX_RADIUS = 150

# --- Helper Functions ---
def create_dummy_frame(text, width, height, text_color=COLOR_RED):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)
    return frame

def calculate_circle_data(detected_circle, screen_center_x, screen_center_y):
    x, y, r = detected_circle
    diff_x = x - screen_center_x
    diff_y = y - screen_center_y
    diff_x_mm = diff_x * MM_PER_PIXEL
    diff_y_mm = diff_y * MM_PER_PIXEL
    distance_mm = (REAL_CIRCLE_RADIUS_MM * FOCAL_LENGTH_MM) / r if r > 0 else 0
    return {'x': x, 'y': y, 'r': r, 'diff_x_mm': diff_x_mm, 'diff_y_mm': diff_y_mm, 'distance_mm': distance_mm}

def draw_overlays(frame, circle_data, screen_center_x, screen_center_y, is_manual_edit=False):
    overlay = frame.copy()
    x, y, r = circle_data['x'], circle_data['y'], circle_data['r']

    cv2.circle(overlay, (x, y), r, COLOR_GREEN, -1)
    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    outline_color = COLOR_YELLOW if is_manual_edit else COLOR_GREEN
    cv2.circle(frame, (x, y), r, outline_color, 2)
    cv2.circle(frame, (x, y), 2, COLOR_RED, 3)
    cv2.circle(frame, (screen_center_x, screen_center_y), 5, COLOR_BLUE, -1)

    text_diff = f"Diff X: {circle_data['diff_x_mm']:.2f}mm, Y: {circle_data['diff_y_mm']:.2f}mm"
    text_dist = f"Distance: {circle_data['distance_mm']:.2f}mm"
    text_radius = f"Radius: {r}px"
    
    cv2.putText(frame, text_diff, (10, 30), FONT, FONT_SCALE, outline_color, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text_dist, (10, 60), FONT, FONT_SCALE, outline_color, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text_radius, (10, 90), FONT, FONT_SCALE, outline_color, FONT_THICKNESS, cv2.LINE_AA)
    
    if is_manual_edit:
        edit_text = "EDIT MODE (w,a,s,d,r,f)"
        cv2.putText(frame, edit_text, (screen_center_x - 150, 30), FONT, FONT_SCALE, COLOR_YELLOW, FONT_THICKNESS, cv2.LINE_AA)

    return frame

def process_frame(frame, history):
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST, param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS)

    if circles is not None:
        history.append(circles[0, 0])
        avg_circle = np.mean(np.array(list(history)), axis=0).astype(int)
        return frame, avg_circle
    else:
        history.clear()
        return frame, None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    screenshot_dir = os.path.join(script_dir, "screenshot")
    os.makedirs(screenshot_dir, exist_ok=True)

    caps = [cv2.VideoCapture(i) for i in range(NUM_CAMERAS)]
    circle_history = [deque(maxlen=HISTORY_SIZE) for _ in range(NUM_CAMERAS)]

    is_paused = False
    manual_edit_mode = False
    editable_camera_info = None
    manual_circle_params = None
    
    original_frames = [None] * NUM_CAMERAS
    last_processed_data = [None] * NUM_CAMERAS
    live_frame = None
    paused_frame = None

    cv2.namedWindow("Combined Camera Feed - Circle Detector", cv2.WINDOW_NORMAL)

    while True:
        if not is_paused:
            processed_frames_for_display = []
            
            for i in range(NUM_CAMERAS):
                if not caps[i].isOpened():
                    ret = False
                    frame = None
                else:
                    ret, frame = caps[i].read()
                
                if not ret:
                    original_frames[i] = None
                    display_frame = create_dummy_frame(f"No Feed Cam {i}", DISPLAY_WIDTH, DISPLAY_HEIGHT)
                    last_processed_data[i] = None
                else:
                    original_frames[i] = frame.copy()
                    processed_frame, detected_circle = process_frame(frame, circle_history[i])
                    
                    if detected_circle is not None:
                        screen_center_x, screen_center_y = processed_frame.shape[1] // 2, processed_frame.shape[0] // 2
                        circle_data = calculate_circle_data(detected_circle, screen_center_x, screen_center_y)
                        circle_data['camera_id'] = i
                        last_processed_data[i] = circle_data
                        display_frame = draw_overlays(processed_frame.copy(), circle_data, screen_center_x, screen_center_y)
                    else:
                        last_processed_data[i] = None
                        display_frame = processed_frame
                
                processed_frames_for_display.append(display_frame)

            row1 = np.hstack(processed_frames_for_display[0:3])
            row2 = np.hstack(processed_frames_for_display[3:6])
            live_frame = np.vstack((row1, row2))
            cv2.imshow("Combined Camera Feed - Circle Detector", live_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '): # Spacebar
            is_paused = not is_paused
            if is_paused:
                if live_frame is not None: paused_frame = live_frame.copy()
                manual_edit_mode = False
                for i, data in enumerate(last_processed_data):
                    if data and original_frames[i] is not None:
                        manual_edit_mode = True
                        editable_camera_info = {'cam_index': i, 'width': original_frames[i].shape[1], 'height': original_frames[i].shape[0]}
                        manual_circle_params = data.copy()
                        print(f"Paused. Manual edit enabled for Camera {i}. Use w,a,s,d to move, r,f to resize.")
                        break
                if not manual_edit_mode:
                    print("Paused. No circle detected to edit.")
            else:
                print("Resumed.")
                manual_edit_mode = False
                editable_camera_info = None
                manual_circle_params = None

        if is_paused and manual_edit_mode:
            adjustment_made = False
            if key == ord('w'):
                manual_circle_params['y'] -= 1; adjustment_made = True
            elif key == ord('s'):
                manual_circle_params['y'] += 1; adjustment_made = True
            elif key == ord('a'):
                manual_circle_params['x'] -= 1; adjustment_made = True
            elif key == ord('d'):
                manual_circle_params['x'] += 1; adjustment_made = True
            elif key == ord('r'):
                manual_circle_params['r'] += 1; adjustment_made = True
            elif key == ord('f'):
                manual_circle_params['r'] = max(1, manual_circle_params['r'] - 1); adjustment_made = True

            if adjustment_made:
                cam_info = editable_camera_info
                screen_center_x, screen_center_y = cam_info['width'] // 2, cam_info['height'] // 2
                updated_data = calculate_circle_data([manual_circle_params['x'], manual_circle_params['y'], manual_circle_params['r']], screen_center_x, screen_center_y)
                manual_circle_params.update(updated_data)

                temp_display_frames = []
                for i in range(NUM_CAMERAS):
                    frame_to_draw = original_frames[i]
                    if frame_to_draw is None:
                        temp_display_frames.append(create_dummy_frame(f"No Feed Cam {i}", DISPLAY_WIDTH, DISPLAY_HEIGHT))
                        continue
                    
                    data_to_draw = last_processed_data[i]
                    is_edit_cam = (i == cam_info['cam_index'])
                    
                    if is_edit_cam:
                        data_to_draw = manual_circle_params

                    if data_to_draw:
                        center_x, center_y = frame_to_draw.shape[1] // 2, frame_to_draw.shape[0] // 2
                        display_frame = draw_overlays(frame_to_draw.copy(), data_to_draw, center_x, center_y, is_manual_edit=is_edit_cam)
                    else:
                        display_frame = cv2.resize(frame_to_draw, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                    temp_display_frames.append(display_frame)

                row1 = np.hstack(temp_display_frames[0:3])
                row2 = np.hstack(temp_display_frames[3:6])
                paused_frame = np.vstack((row1, row2))
                cv2.imshow("Combined Camera Feed - Circle Detector", paused_frame)

        if key == ord('s'):
            frame_to_save = paused_frame if is_paused and paused_frame is not None else live_frame
            if frame_to_save is None:
                print("No frame to save.")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_filename = os.path.join(script_dir, f"screenshot_{timestamp}.png")
            excel_filepath = os.path.join(script_dir, MASTER_EXCEL_FILENAME)

            data_to_save = list(filter(None, last_processed_data))
            
            if is_paused and manual_edit_mode and manual_circle_params:
                cam_idx_to_update = editable_camera_info['cam_index']
                found = False
                for i, record in enumerate(data_to_save):
                    if record['camera_id'] == cam_idx_to_update:
                        manual_circle_params['camera_id'] = cam_idx_to_update
                        data_to_save[i] = manual_circle_params
                        found = True
                        break
                if not found:
                    manual_circle_params['camera_id'] = cam_idx_to_update
                    data_to_save.append(manual_circle_params)

            cv2.imwrite(screenshot_filename, frame_to_save)
            print(f"Screenshot saved as {screenshot_filename}")

            if data_to_save:
                for record in data_to_save:
                    record['timestamp'] = timestamp
                
                new_data_df = pd.DataFrame(data_to_save)
                
                try:
                    if os.path.exists(excel_filepath):
                        existing_df = pd.read_excel(excel_filepath)
                        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
                    else:
                        combined_df = new_data_df
                    
                    cols = ['timestamp', 'camera_id', 'x', 'y', 'r', 'diff_x_mm', 'diff_y_mm', 'distance_mm']
                    existing_cols = [col for col in cols if col in combined_df.columns]
                    combined_df = combined_df[existing_cols]

                    combined_df.to_excel(excel_filepath, index=False)
                    print(f"Data appended to {excel_filepath}")
                except Exception as e:
                    print(f"Could not save or append to Excel file: {e}")
            else:
                print("No circle data to save.")

        if key == ord('q'):
            break

    for cap in caps:
        if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()