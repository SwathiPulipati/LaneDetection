import cv2
import numpy as np

vid = cv2.VideoCapture("C:\\Users\\Swathi Pulipati\\Swathi\\Comp Sci\\is\\road\\road.mp4")


def region(image):
    height, width, _ = image.shape
    polygon = np.array([[(-400, height), (920, 560), (1125, 550), (1790, height)]], np.int32)
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, polygon, (255, 255, 255, 0))
    mask = cv2.bitwise_and(image, mask)
    return mask


def make_points(image, average):
    try:
        slope, y_int = average
        y1 = image.shape[0]
        y2 = int(y1 * (11/20))
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
        return np.array([x1, y1, x2, y2])
    except TypeError:
        return None


def average(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            right.append((slope, y_int))
        else:
            left.append((slope, y_int))
    right_avg = np.average(right, 0)
    left_avg = np.average(left, 0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            try:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0, 1), 10)
            except:
                continue
        try:
            coords = np.reshape(lines, (4, 2))
            points = coords[2]
            coords = np.delete(coords, 2, 0)
            coords = np.append(coords, points)
            coords = np.reshape(coords, (4, 2))
            lines_image = cv2.fillPoly(lines_image, np.int32([coords]), (35, 200, 0))
        except ValueError:
            None
    return lines_image


while vid.isOpened():
    frame = vid.read()[1]
    cropped_mask = region(frame)
    masked = cv2.addWeighted(frame, .4, cropped_mask, .5, 0)

    hsv_frame = cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2HSV)
    l_yellow = np.array([20, 100, 100], "uint8")
    u_yellow = np.array([30, 255, 255], "uint8")
    mask_yellow = cv2.inRange(hsv_frame, l_yellow, u_yellow)
    gray = cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2GRAY)
    mask_white = cv2.inRange(gray, 200, 255)
    w_y_mask = cv2.bitwise_or(mask_white, mask_yellow)

    blur = cv2.GaussianBlur(w_y_mask, (3, 3), 0)
    edges = cv2.Canny(blur, 100, 200)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 40, 20)
    avg_lines = average(cropped_mask, lines)
    black_lines = display_lines(cropped_mask, avg_lines)
    lanes = cv2.addWeighted(masked, 1, black_lines, 1, 0)

    # cv2.imshow('og', frame)
    # cv2.imshow('cropped', cropped_mask)
    # cv2.imshow('threshold colors', w_y_mask)
    # cv2.imshow('edges', edges)
    cv2.imshow('final', lanes)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
vid.release()
cv2.destroyAllWindows()


