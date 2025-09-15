import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import os

# --- Constants ---
NUM_CAMERAS = 6
MM_PER_PIXEL = 0.1
REAL_ELLIPSE_MAJOR_AXIS_MM = 6.4  # Assuming this is the major axis of the real ellipse
FOCAL_LENGTH_MM = 3910.31
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
HISTORY_SIZE = 5
MASTER_EXCEL_FILENAME = "ellipse_log.xlsx"

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

# --- Helper Functions ---
def create_dummy_frame(text, width, height, text_color=COLOR_RED):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)
    return frame

def calculate_ellipse_data(detected_ellipse, screen_center_x, screen_center_y):
    (x, y), (MA, ma), angle = detected_ellipse
    x, y, MA, ma = int(x), int(y), int(MA), int(ma)
    diff_x = x - screen_center_x
    diff_y = y - screen_center_y
    diff_x_mm = diff_x * MM_PER_PIXEL
    diff_y_mm = diff_y * MM_PER_PIXEL
    distance_mm = (REAL_ELLIPSE_MAJOR_AXIS_MM * FOCAL_LENGTH_MM) / MA if MA > 0 else 0
    return {'x': x, 'y': y, 'MA': MA, 'ma': ma, 'angle': angle, 'diff_x_mm': diff_x_mm, 'diff_y_mm': diff_y_mm, 'distance_mm': distance_mm}

def draw_overlays(frame, ellipse_data, screen_center_x, screen_center_y, is_manual_edit=False):
    overlay = frame.copy()
    x, y, MA, ma, angle = ellipse_data['x'], ellipse_data['y'], ellipse_data['MA'], ellipse_data['ma'], ellipse_data['angle']

    cv2.ellipse(overlay, (x, y), (MA // 2, ma // 2), angle, 0, 360, COLOR_GREEN, -1)
    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    outline_color = COLOR_YELLOW if is_manual_edit else COLOR_GREEN
    cv2.ellipse(frame, (x, y), (MA // 2, ma // 2), angle, 0, 360, outline_color, 2)
    cv2.circle(frame, (x, y), 2, COLOR_RED, 3)
    cv2.circle(frame, (screen_center_x, screen_center_y), 5, COLOR_BLUE, -1)

    # Draw major and minor axes
    angle_rad = np.deg2rad(angle)
    major_axis_p1 = (int(x + 0.5 * MA/2 * np.cos(angle_rad)), int(y + 0.5 * MA/2 * np.sin(angle_rad)))
    major_axis_p2 = (int(x - 0.5 * MA/2 * np.cos(angle_rad)), int(y - 0.5 * MA/2 * np.sin(angle_rad)))
    cv2.line(frame, major_axis_p1, major_axis_p2, COLOR_RED, 2)

    minor_axis_p1 = (int(x - 0.5 * ma/2 * np.sin(angle_rad)), int(y + 0.5 * ma/2 * np.cos(angle_rad)))
    minor_axis_p2 = (int(x + 0.5 * ma/2 * np.sin(angle_rad)), int(y - 0.5 * ma/2 * np.cos(angle_rad)))
    cv2.line(frame, minor_axis_p1, minor_axis_p2, COLOR_BLUE, 2)

    text_diff = f"Diff X: {ellipse_data['diff_x_mm']:.2f}mm, Y: {ellipse_data['diff_y_mm']:.2f}mm"
    text_dist = f"Distance: {ellipse_data['distance_mm']:.2f}mm"
    text_axes = f"Axes: {MA}px, {ma}px"
    
    cv2.putText(frame, text_diff, (10, 30), FONT, FONT_SCALE, outline_color, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text_dist, (10, 60), FONT, FONT_SCALE, outline_color, FONT_THICKNESS, cv2.LINE_AA)
    cv2.putText(frame, text_axes, (10, 90), FONT, FONT_SCALE, outline_color, FONT_THICKNESS, cv2.LINE_AA)
    
    if is_manual_edit:
        edit_text = "EDIT MODE (w,a,s,d,r,f,t,g)"
        cv2.putText(frame, edit_text, (screen_center_x - 200, 30), FONT, FONT_SCALE, COLOR_YELLOW, FONT_THICKNESS, cv2.LINE_AA)

    return frame

def process_frame(frame, history):
    frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ellipse = None
    max_area = 0

    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
            if area > max_area:
                max_area = area
                best_ellipse = ellipse

    if best_ellipse is not None:
        history.append(best_ellipse)
        # Averaging ellipse parameters is complex, so we'll just use the latest detection for now
        return frame, best_ellipse
    else:
        history.clear()
        return frame, None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    screenshot_dir = os.path.join(script_dir, "screenshot")
    os.makedirs(screenshot_dir, exist_ok=True)

    caps = [cv2.VideoCapture(i) for i in range(NUM_CAMERAS)]
    ellipse_history = [deque(maxlen=HISTORY_SIZE) for _ in range(NUM_CAMERAS)]

    is_paused = False
    manual_edit_mode = False
    editable_camera_info = None
    manual_ellipse_params = None
    
    original_frames = [None] * NUM_CAMERAS
    last_processed_data = [None] * NUM_CAMERAS
    live_frame = None
    paused_frame = None

    cv2.namedWindow("Combined Camera Feed - Ellipse Detector", cv2.WINDOW_NORMAL)

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
                    processed_frame, detected_ellipse = process_frame(frame, ellipse_history[i])
                    
                    if detected_ellipse is not None:
                        screen_center_x, screen_center_y = processed_frame.shape[1] // 2, processed_frame.shape[0] // 2
                        ellipse_data = calculate_ellipse_data(detected_ellipse, screen_center_x, screen_center_y)
                        ellipse_data['camera_id'] = i
                        last_processed_data[i] = ellipse_data
                        display_frame = draw_overlays(processed_frame.copy(), ellipse_data, screen_center_x, screen_center_y)
                    else:
                        last_processed_data[i] = None
                        display_frame = processed_frame
                
                processed_frames_for_display.append(display_frame)

            row1 = np.hstack(processed_frames_for_display[0:3])
            row2 = np.hstack(processed_frames_for_display[3:6])
            live_frame = np.vstack((row1, row2))
            cv2.imshow("Combined Camera Feed - Ellipse Detector", live_frame)

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
                        manual_ellipse_params = data.copy()
                        print(f"Paused. Manual edit enabled for Camera {i}. Use w,a,s,d to move, r,f for major axis, t,g for minor axis.")
                        break
                if not manual_edit_mode:
                    print("Paused. No ellipse detected to edit.")
            else:
                print("Resumed.")
                manual_edit_mode = False
                editable_camera_info = None
                manual_ellipse_params = None

        if is_paused and manual_edit_mode:
            adjustment_made = False
            if key == ord('w'):
                manual_ellipse_params['y'] -= 1; adjustment_made = True
            elif key == ord('s'):
                manual_ellipse_params['y'] += 1; adjustment_made = True
            elif key == ord('a'):
                manual_ellipse_params['x'] -= 1; adjustment_made = True
            elif key == ord('d'):
                manual_ellipse_params['x'] += 1; adjustment_made = True
            elif key == ord('r'):
                manual_ellipse_params['MA'] += 1; adjustment_made = True
            elif key == ord('f'):
                manual_ellipse_params['MA'] = max(1, manual_ellipse_params['MA'] - 1); adjustment_made = True
            elif key == ord('t'):
                manual_ellipse_params['ma'] += 1; adjustment_made = True
            elif key == ord('g'):
                manual_ellipse_params['ma'] = max(1, manual_ellipse_params['ma'] - 1); adjustment_made = True

            if adjustment_made:
                cam_info = editable_camera_info
                screen_center_x, screen_center_y = cam_info['width'] // 2, cam_info['height'] // 2
                updated_data = calculate_ellipse_data(((manual_ellipse_params['x'], manual_ellipse_params['y']), (manual_ellipse_params['MA'], manual_ellipse_params['ma']), manual_ellipse_params['angle']), screen_center_x, screen_center_y)
                manual_ellipse_params.update(updated_data)

                temp_display_frames = []
                for i in range(NUM_CAMERAS):
                    frame_to_draw = original_frames[i]
                    if frame_to_draw is None:
                        temp_display_frames.append(create_dummy_frame(f"No Feed Cam {i}", DISPLAY_WIDTH, DISPLAY_HEIGHT))
                        continue
                    
                    data_to_draw = last_processed_data[i]
                    is_edit_cam = (i == cam_info['cam_index'])
                    
                    if is_edit_cam:
                        data_to_draw = manual_ellipse_params

                    if data_to_draw:
                        center_x, center_y = frame_to_draw.shape[1] // 2, frame_to_draw.shape[0] // 2
                        display_frame = draw_overlays(frame_to_draw.copy(), data_to_draw, center_x, center_y, is_manual_edit=is_edit_cam)
                    else:
                        display_frame = cv2.resize(frame_to_draw, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                    temp_display_frames.append(display_frame)

                row1 = np.hstack(temp_display_frames[0:3])
                row2 = np.hstack(temp_display_frames[3:6])
                paused_frame = np.vstack((row1, row2))
                cv2.imshow("Combined Camera Feed - Ellipse Detector", paused_frame)

        if key == ord('s'):
            frame_to_save = paused_frame if is_paused and paused_frame is not None else live_frame
            if frame_to_save is None:
                print("No frame to save.")
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_filename = os.path.join(script_dir, f"screenshot_{timestamp}.png")
            excel_filepath = os.path.join(script_dir, MASTER_EXCEL_FILENAME)

            data_to_save = list(filter(None, last_processed_data))
            
            if is_paused and manual_edit_mode and manual_ellipse_params:
                cam_idx_to_update = editable_camera_info['cam_index']
                found = False
                for i, record in enumerate(data_to_save):
                    if record['camera_id'] == cam_idx_to_update:
                        manual_ellipse_params['camera_id'] = cam_idx_to_update
                        data_to_save[i] = manual_ellipse_params
                        found = True
                        break
                if not found:
                    manual_ellipse_params['camera_id'] = cam_idx_to_update
                    data_to_save.append(manual_ellipse_params)

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
                    
                    cols = ['timestamp', 'camera_id', 'x', 'y', 'MA', 'ma', 'angle', 'diff_x_mm', 'diff_y_mm', 'distance_mm']
                    existing_cols = [col for col in cols if col in combined_df.columns]
                    combined_df = combined_df[existing_cols]

                    combined_df.to_excel(excel_filepath, index=False)
                    print(f"Data appended to {excel_filepath}")
                except Exception as e:
                    print(f"Could not save or append to Excel file: {e}")
            else:
                print("No ellipse data to save.")

        if key == ord('q'):
            break

    for cap in caps:
        if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()