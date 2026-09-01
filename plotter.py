import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import threading
import time
import math


# ============================================================
# DEFAULT SETTINGS
# ============================================================

BAUD_RATE = 115200

DEFAULT_X_STEPS_MM = 21.28
DEFAULT_Y_STEPS_MM = 20.95

DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100

DEFAULT_PEN_UP = 90
DEFAULT_PEN_DOWN = 30

DEFAULT_THRESHOLD = 150

DEFAULT_MIN_CONTOUR_LENGTH = 15
DEFAULT_MIN_CONTOUR_AREA = 5
DEFAULT_MIN_PATH_LENGTH = 1.5

DEFAULT_SIMPLIFY_EPSILON = 0.8

DEFAULT_MORPH_KERNEL_SIZE = 3

DEFAULT_MIN_POINT_DISTANCE = 0.2

DEFAULT_EDGE_LOW = 50
DEFAULT_EDGE_HIGH = 150

DEFAULT_BLUR_SIZE = 3

DEFAULT_MIN_STROKE_LENGTH = 2.0

DEFAULT_MAX_PATHS = 2000

DEFAULT_DUPLICATE_TOLERANCE = 0.5


# ============================================================
# GLOBALS
# ============================================================

image_original = None
image_processed = None

toolpaths = []

ser = None
plotting = False


# ============================================================
# COLLAPSIBLE FRAME
# ============================================================

class CollapsibleFrame(tk.Frame):

    def __init__(
        self,
        parent,
        title,
        expanded=True
    ):

        super().__init__(
            parent,
            bd=1,
            relief="solid"
        )

        self.expanded = expanded

        self.header = tk.Frame(
            self
        )

        self.header.pack(
            fill="x"
        )

        self.toggle_button = tk.Button(
            self.header,
            text=("▼ " if expanded else "▶ ") + title,
            anchor="w",
            relief="flat",
            bd=0,
            font=("Arial", 10, "bold"),
            command=self.toggle
        )

        self.toggle_button.pack(
            fill="x",
            padx=4,
            pady=3
        )

        self.content = tk.Frame(
            self
        )

        if expanded:

            self.content.pack(
                fill="x",
                padx=8,
                pady=6
            )

    def toggle(self):

        if self.expanded:

            self.content.pack_forget()

            self.toggle_button.config(
                text="▶ " +
                self.toggle_button.cget("text")[2:]
            )

            self.expanded = False

        else:

            self.content.pack(
                fill="x",
                padx=8,
                pady=6
            )

            self.toggle_button.config(
                text="▼ " +
                self.toggle_button.cget("text")[2:]
            )

            self.expanded = True

        left_canvas.update_idletasks()

        left_canvas.configure(
            scrollregion=
            left_canvas.bbox("all")
        )


# ============================================================
# HELPER
# ============================================================

def safe_int(value, default):

    try:
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default):

    try:
        return float(value)
    except Exception:
        return default


# ============================================================
# IMAGE
# ============================================================

def open_image():

    global image_original
    global image_processed
    global toolpaths

    filename = filedialog.askopenfilename(
        filetypes=[
            (
                "Image files",
                "*.png *.jpg *.jpeg *.bmp *.webp"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )

    if not filename:
        return

    img = cv2.imread(
        filename
    )

    if img is None:

        messagebox.showerror(
            "Error",
            "Could not load image."
        )

        return

    image_original = img
    image_processed = None
    toolpaths = []

    show_image(
        img
    )

    process_image()

    canvas.delete(
        "all"
    )

    status_var.set(
        "Loaded: " + filename
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image():

    global image_processed

    if image_original is None:
        return

    img = image_original.copy()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur_size = safe_int(
        blur_var.get(),
        DEFAULT_BLUR_SIZE
    )

    if blur_size < 1:
        blur_size = 1

    if blur_size % 2 == 0:
        blur_size += 1

    if blur_size > 1:

        gray = cv2.GaussianBlur(
            gray,
            (
                blur_size,
                blur_size
            ),
            0
        )

    mode = processing_mode.get()

    threshold = safe_int(
        threshold_var.get(),
        DEFAULT_THRESHOLD
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    if mode == "Threshold":

        _, processed = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

    # ========================================================
    # ADAPTIVE
    # ========================================================

    elif mode == "Adaptive":

        block_size = safe_int(
            adaptive_block_var.get(),
            11
        )

        if block_size < 3:
            block_size = 3

        if block_size % 2 == 0:
            block_size += 1

        c_value = safe_int(
            adaptive_c_var.get(),
            2
        )

        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value
        )

    # ========================================================
    # CANNY
    # ========================================================

    elif mode == "Edge Detection":

        low = safe_int(
            edge_low_var.get(),
            DEFAULT_EDGE_LOW
        )

        high = safe_int(
            edge_high_var.get(),
            DEFAULT_EDGE_HIGH
        )

        if high <= low:
            high = low + 10

        edges = cv2.Canny(
            gray,
            low,
            high
        )

        processed = edges

    else:

        processed = gray

    image_processed = processed

    show_image(
        processed
    )


# ============================================================
# SHOW IMAGE
# ============================================================

def show_image(img):

    if img is None:
        return

    display = img.copy()

    if len(display.shape) == 2:

        display = cv2.cvtColor(
            display,
            cv2.COLOR_GRAY2RGB
        )

    else:

        display = cv2.cvtColor(
            display,
            cv2.COLOR_BGR2RGB
        )

    pil = Image.fromarray(
        display
    )

    pil.thumbnail(
        (620, 320)
    )

    photo = ImageTk.PhotoImage(
        pil
    )

    image_label.config(
        image=photo
    )

    image_label.image = photo


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return math.hypot(
        b[0] - a[0],
        b[1] - a[1]
    )


# ============================================================
# REMOVE CLOSE POINTS
# ============================================================

def remove_close_points(
    path,
    min_distance
):

    if len(path) < 2:
        return path

    cleaned = [
        path[0]
    ]

    for point in path[1:]:

        if distance(
            cleaned[-1],
            point
        ) >= min_distance:

            cleaned.append(
                point
            )

    return cleaned


# ============================================================
# PATH LENGTH
# ============================================================

def calculate_path_length(
    path,
    closed=False
):

    if len(path) < 2:
        return 0

    total = 0

    for i in range(
        1,
        len(path)
    ):

        total += distance(
            path[i - 1],
            path[i]
        )

    if closed and len(path) >= 3:

        total += distance(
            path[-1],
            path[0]
        )

    return total


# ============================================================
# REVERSE
# ============================================================

def reverse_path(path):

    return list(
        reversed(path)
    )


# ============================================================
# PATH DUPLICATE FILTER
# ============================================================

def remove_duplicate_paths(
    paths,
    tolerance
):

    if not paths:
        return []

    unique = []

    for path in paths:

        if len(path) < 2:
            continue

        xs = [
            p[0]
            for p in path
        ]

        ys = [
            p[1]
            for p in path
        ]

        bbox = (
            min(xs),
            min(ys),
            max(xs),
            max(ys)
        )

        duplicate = False

        for existing in unique:

            ex = [
                p[0]
                for p in existing
            ]

            ey = [
                p[1]
                for p in existing
            ]

            ebbox = (
                min(ex),
                min(ey),
                max(ex),
                max(ey)
            )

            if all(
                abs(
                    bbox[i] -
                    ebbox[i]
                ) <= tolerance
                for i in range(4)
            ):

                duplicate = True
                break

        if not duplicate:

            unique.append(
                path
            )

    return unique


# ============================================================
# PATH OPTIMIZATION
# ============================================================

def optimize_paths(paths):

    if not paths:
        return []

    remaining = [
        list(path)
        for path in paths
    ]

    optimized = []

    current = (
        0,
        0
    )

    while remaining:

        best_index = 0
        best_distance = float("inf")
        best_reverse = False

        for i, path in enumerate(
            remaining
        ):

            if len(path) < 2:
                continue

            d_start = distance(
                current,
                path[0]
            )

            d_end = distance(
                current,
                path[-1]
            )

            if d_start < best_distance:

                best_distance = d_start
                best_index = i
                best_reverse = False

            if d_end < best_distance:

                best_distance = d_end
                best_index = i
                best_reverse = True

        path = remaining.pop(
            best_index
        )

        if best_reverse:

            path = reverse_path(
                path
            )

        optimized.append(
            path
        )

        current = path[-1]

    return optimized


# ============================================================
# MORPHOLOGICAL CLEANUP
# ============================================================

def clean_binary(
    binary,
    kernel_size
):

    if kernel_size <= 1:
        return binary

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        (
            kernel_size,
            kernel_size
        ),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return binary


# ============================================================
# TRACE VECTOR PATHS
#
# PHOTO -> MANY SVG-LIKE PATHS
# ============================================================

def generate_trace_paths(
    binary,
    width_mm,
    height_mm,
    simplify_epsilon,
    min_contour_length,
    min_contour_area,
    min_path_length,
    min_point_distance,
    morph_kernel_size,
    max_paths,
    closed_contours
):

    # --------------------------------------------------------
    # Convert image to drawing mask
    # --------------------------------------------------------

    drawing = binary.copy()

    _, drawing = cv2.threshold(
        drawing,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    drawing = clean_binary(
        drawing,
        morph_kernel_size
    )

    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        drawing,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return []

    h, w = drawing.shape

    scale_x = width_mm / w
    scale_y = height_mm / h

    paths = []

    # --------------------------------------------------------
    # CONTOURS -> VECTOR
    # --------------------------------------------------------

    for contour in contours:

        if len(contour) < 3:
            continue

        area = abs(
            cv2.contourArea(
                contour
            )
        )

        if area < min_contour_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            closed_contours
        )

        if perimeter < min_contour_length:
            continue

        # epsilon is in pixels here
        epsilon_px = (
            simplify_epsilon
            / max(
                scale_x,
                scale_y
            )
        )

        simplified = cv2.approxPolyDP(
            contour,
            epsilon_px,
            closed_contours
        )

        if len(simplified) < 2:
            continue

        path = []

        for p in simplified:

            x = float(
                p[0][0]
            )

            y = float(
                p[0][1]
            )

            path.append(
                (
                    x * scale_x,
                    y * scale_y
                )
            )

        path = remove_close_points(
            path,
            min_point_distance
        )

        if len(path) < 2:
            continue

        length = calculate_path_length(
            path,
            closed=closed_contours
        )

        if length < min_path_length:
            continue

        paths.append(
            path
        )

    # --------------------------------------------------------
    # LONGEST FIRST
    # --------------------------------------------------------

    paths.sort(
        key=calculate_path_length,
        reverse=True
    )

    if max_paths > 0:

        paths = paths[
            :max_paths
        ]

    return paths


# ============================================================
# SKELETONIZATION
# ============================================================

def skeletonize(binary):

    """
    Morphological skeletonization.

    No skimage dependency required.
    """

    img = binary.copy()

    _, img = cv2.threshold(
        img,
        127,
        255,
        cv2.THRESH_BINARY
    )

    skeleton = np.zeros_like(
        img
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (
            3,
            3
        )
    )

    while True:

        eroded = cv2.erode(
            img,
            kernel
        )

        opened = cv2.dilate(
            eroded,
            kernel
        )

        temp = cv2.subtract(
            img,
            opened
        )

        skeleton = cv2.bitwise_or(
            skeleton,
            temp
        )

        img = eroded.copy()

        if cv2.countNonZero(
            img
        ) == 0:

            break

    return skeleton


# ============================================================
# MEANINGFUL STROKES
#
# PHOTO -> EDGES -> STROKES
# ============================================================

def generate_stroke_paths(
    gray,
    width_mm,
    height_mm,
    edge_low,
    edge_high,
    simplify_epsilon,
    min_stroke_length,
    min_point_distance,
    morph_kernel_size,
    max_paths
):

    # --------------------------------------------------------
    # EDGE DETECTION
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        edge_low,
        edge_high
    )

    # --------------------------------------------------------
    # CLEAN EDGES
    # --------------------------------------------------------

    if morph_kernel_size > 1:

        kernel_size = morph_kernel_size

        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(
            (
                kernel_size,
                kernel_size
            ),
            np.uint8
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel
        )

    # --------------------------------------------------------
    # SKELETONIZE
    # --------------------------------------------------------

    skeleton = skeletonize(
        edges
    )

    # --------------------------------------------------------
    # FIND OPEN STROKES
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        skeleton,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return []

    h, w = skeleton.shape

    scale_x = width_mm / w
    scale_y = height_mm / h

    paths = []

    # --------------------------------------------------------
    # CONVERT TO VECTOR PATHS
    # --------------------------------------------------------

    for contour in contours:

        if len(contour) < 2:
            continue

        perimeter = cv2.arcLength(
            contour,
            False
        )

        length_mm = (
            perimeter *
            (scale_x + scale_y) /
            2
        )

        if length_mm < min_stroke_length:
            continue

        epsilon_px = (
            simplify_epsilon /
            max(
                scale_x,
                scale_y
            )
        )

        simplified = cv2.approxPolyDP(
            contour,
            epsilon_px,
            False
        )

        if len(simplified) < 2:
            continue

        path = []

        for p in simplified:

            x = float(
                p[0][0]
            )

            y = float(
                p[0][1]
            )

            path.append(
                (
                    x * scale_x,
                    y * scale_y
                )
            )

        path = remove_close_points(
            path,
            min_point_distance
        )

        if len(path) < 2:
            continue

        if calculate_path_length(
            path
        ) < min_stroke_length:

            continue

        paths.append(
            path
        )

    # --------------------------------------------------------
    # LONGEST FIRST
    # --------------------------------------------------------

    paths.sort(
        key=calculate_path_length,
        reverse=True
    )

    if max_paths > 0:

        paths = paths[
            :max_paths
        ]

    return paths


# ============================================================
# GENERATE TOOLPATHS
# ============================================================

def generate_toolpaths():

    global toolpaths

    if image_original is None:

        messagebox.showwarning(
            "Warning",
            "Open an image first."
        )

        return

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    try:

        width_mm = float(
            width_var.get()
        )

        height_mm = float(
            height_var.get()
        )

        simplify_epsilon = float(
            simplify_var.get()
        )

        min_contour_length = float(
            min_contour_length_var.get()
        )

        min_contour_area = float(
            min_contour_area_var.get()
        )

        min_path_length = float(
            min_path_length_var.get()
        )

        min_point_distance = float(
            min_point_distance_var.get()
        )

        morph_kernel_size = int(
            morph_kernel_var.get()
        )

        max_paths = int(
            max_paths_var.get()
        )

        duplicate_tolerance = float(
            duplicate_tolerance_var.get()
        )

        min_stroke_length = float(
            min_stroke_length_var.get()
        )

        edge_low = int(
            edge_low_var.get()
        )

        edge_high = int(
            edge_high_var.get()
        )

    except ValueError:

        messagebox.showerror(
            "Error",
            "One or more settings contain invalid numbers."
        )

        return

    if width_mm <= 0 or height_mm <= 0:

        messagebox.showerror(
            "Error",
            "Plot dimensions must be greater than zero."
        )

        return

    if simplify_epsilon < 0:

        messagebox.showerror(
            "Error",
            "Simplify epsilon cannot be negative."
        )

        return

    if min_contour_length < 0:
        messagebox.showerror(
            "Error",
            "Minimum contour length cannot be negative."
        )
        return

    if min_contour_area < 0:
        messagebox.showerror(
            "Error",
            "Minimum contour area cannot be negative."
        )
        return

    if min_path_length < 0:
        messagebox.showerror(
            "Error",
            "Minimum path length cannot be negative."
        )
        return

    if min_point_distance < 0:
        messagebox.showerror(
            "Error",
            "Minimum point distance cannot be negative."
        )
        return

    if morph_kernel_size < 1:
        morph_kernel_size = 1

    if morph_kernel_size % 2 == 0:
        morph_kernel_size += 1

    # --------------------------------------------------------
    # PREPARE IMAGE
    # --------------------------------------------------------

    img = image_original.copy()

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    blur_size = int(
        blur_var.get()
    )

    if blur_size < 1:
        blur_size = 1

    if blur_size % 2 == 0:
        blur_size += 1

    if blur_size > 1:

        gray = cv2.GaussianBlur(
            gray,
            (
                blur_size,
                blur_size
            ),
            0
        )

    vector_mode = vectorization_mode.get()

    # ========================================================
    # METHOD 1:
    # SVG TRACE / MANY PATHS
    # ========================================================

    if vector_mode == "SVG Trace - Many Paths":

        process_mode = processing_mode.get()

        if process_mode == "Threshold":

            _, processed = cv2.threshold(
                gray,
                threshold_var.get(),
                255,
                cv2.THRESH_BINARY
            )

        elif process_mode == "Adaptive":

            block_size = int(
                adaptive_block_var.get()
            )

            if block_size < 3:
                block_size = 3

            if block_size % 2 == 0:
                block_size += 1

            c_value = int(
                adaptive_c_var.get()
            )

            processed = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_value
            )

        else:

            processed = cv2.Canny(
                gray,
                edge_low,
                edge_high
            )

        toolpaths = generate_trace_paths(
            processed,
            width_mm,
            height_mm,
            simplify_epsilon,
            min_contour_length,
            min_contour_area,
            min_path_length,
            min_point_distance,
            morph_kernel_size,
            max_paths,
            closed_contours_var.get()
        )

    # ========================================================
    # METHOD 2:
    # MEANINGFUL STROKES
    # ========================================================

    else:

        toolpaths = generate_stroke_paths(
            gray,
            width_mm,
            height_mm,
            edge_low,
            edge_high,
            simplify_epsilon,
            min_stroke_length,
            min_point_distance,
            morph_kernel_size,
            max_paths
        )

    # --------------------------------------------------------
    # DUPLICATE FILTER
    # --------------------------------------------------------

    toolpaths = remove_duplicate_paths(
        toolpaths,
        duplicate_tolerance
    )

    # --------------------------------------------------------
    # FINAL SORT / LIMIT
    # --------------------------------------------------------

    toolpaths.sort(
        key=calculate_path_length,
        reverse=True
    )

    if max_paths > 0:

        toolpaths = toolpaths[
            :max_paths
        ]

    # --------------------------------------------------------
    # OPTIMIZE
    # --------------------------------------------------------

    if optimize_paths_var.get():

        toolpaths = optimize_paths(
            toolpaths
        )

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    draw_toolpath_preview()

    total_length = sum(
        calculate_path_length(p)
        for p in toolpaths
    )

    status_var.set(
        f"{len(toolpaths)} paths | "
        f"{total_length:.1f} mm"
    )


# ============================================================
# SVG EXPORT
# ============================================================

def export_svg():

    if not toolpaths:

        messagebox.showwarning(
            "Warning",
            "Generate toolpaths first."
        )

        return

    filename = filedialog.asksaveasfilename(
        defaultextension=".svg",
        filetypes=[
            (
                "SVG files",
                "*.svg"
            )
        ]
    )

    if not filename:
        return

    try:

        width = float(
            width_var.get()
        )

        height = float(
            height_var.get()
        )

        svg = []

        svg.append(
            '<?xml version="1.0" encoding="UTF-8"?>'
        )

        svg.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}mm" '
            f'height="{height}mm" '
            f'viewBox="0 0 {width} {height}">'
        )

        svg.append(
            '<g fill="none" stroke="black" '
            'stroke-width="0.25" '
            'stroke-linecap="round" '
            'stroke-linejoin="round">'
        )

        for path in toolpaths:

            if len(path) < 2:
                continue

            d = (
                f"M {path[0][0]:.3f} "
                f"{path[0][1]:.3f} "
            )

            for x, y in path[1:]:

                d += (
                    f"L {x:.3f} "
                    f"{y:.3f} "
                )

            if (
                vectorization_mode.get()
                ==
                "SVG Trace - Many Paths"
                and
                closed_contours_var.get()
            ):

                d += "Z"

            svg.append(
                f'<path d="{d}"/>'
            )

        svg.append(
            "</g>"
        )

        svg.append(
            "</svg>"
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(svg)
            )

        status_var.set(
            "SVG exported"
        )

    except Exception as e:

        messagebox.showerror(
            "SVG Export Error",
            str(e)
        )


# ============================================================
# PREVIEW
# ============================================================

def draw_toolpath_preview():

    canvas.delete(
        "all"
    )

    if not toolpaths:
        return

    try:

        width = float(
            width_var.get()
        )

        height = float(
            height_var.get()
        )

    except ValueError:

        return

    canvas_width = 650
    canvas_height = 500

    scale = min(
        canvas_width / width,
        canvas_height / height
    )

    offset_x = (
        canvas_width -
        width * scale
    ) / 2

    offset_y = (
        canvas_height -
        height * scale
    ) / 2

    # --------------------------------------------------------
    # BORDER
    # --------------------------------------------------------

    canvas.create_rectangle(
        offset_x,
        offset_y,
        offset_x +
        width * scale,
        offset_y +
        height * scale,
        outline="gray"
    )

    previous_end = None

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    for path in toolpaths:

        if len(path) < 2:
            continue

        # ----------------------------------------------------
        # PEN-UP TRAVEL
        # ----------------------------------------------------

        if previous_end is not None:

            start = path[0]

            x1 = (
                offset_x +
                previous_end[0] *
                scale
            )

            y1 = (
                offset_y +
                previous_end[1] *
                scale
            )

            x2 = (
                offset_x +
                start[0] *
                scale
            )

            y2 = (
                offset_y +
                start[1] *
                scale
            )

            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                dash=(3, 3),
                fill="gray"
            )

        # ----------------------------------------------------
        # DRAW PATH
        # ----------------------------------------------------

        points = []

        for x, y in path:

            points.extend(
                [
                    offset_x +
                    x * scale,

                    offset_y +
                    y * scale
                ]
            )

        if len(points) >= 4:

            canvas.create_line(
                *points,
                width=1
            )

        previous_end = path[-1]


# ============================================================
# SERIAL
# ============================================================

def refresh_ports():

    ports = serial.tools.list_ports.comports()

    port_combo["values"] = [
        p.device
        for p in ports
    ]

    if ports:

        port_combo.current(
            0
        )


def connect_serial():

    global ser

    port = port_var.get()

    if not port:

        messagebox.showwarning(
            "Warning",
            "Select a COM port."
        )

        return

    try:

        if ser is not None:

            try:
                ser.close()
            except Exception:
                pass

        ser = serial.Serial(
            port,
            BAUD_RATE,
            timeout=5
        )

        time.sleep(
            2
        )

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        status_var.set(
            f"Connected to {port}"
        )

        connect_button.config(
            text="Connected"
        )

    except Exception as e:

        ser = None

        messagebox.showerror(
            "Connection Error",
            str(e)
        )


# ============================================================
# SEND COMMAND
# ============================================================

def send_command(command):

    if ser is None:
        return False

    try:

        ser.write(
            (
                command +
                "\n"
            ).encode()
        )

        ser.flush()

        ser.readline()

        return True

    except Exception:

        return False


# ============================================================
# STEPS
# ============================================================

def mm_to_steps_x(mm):

    return round(
        mm *
        float(
            x_steps_var.get()
        )
    )


def mm_to_steps_y(mm):

    return round(
        mm *
        float(
            y_steps_var.get()
        )
    )


# ============================================================
# START PLOT
# ============================================================

def start_plot():

    global plotting

    if ser is None:

        messagebox.showwarning(
            "Warning",
            "Connect Arduino first."
        )

        return

    if not toolpaths:

        messagebox.showwarning(
            "Warning",
            "Generate toolpaths first."
        )

        return

    if plotting:
        return

    answer = messagebox.askyesno(
        "Start Plot",
        "Make sure the pen is positioned at HOME.\n\n"
        "Start plotting?"
    )

    if not answer:
        return

    plotting = True

    plot_button.config(
        state="disabled"
    )

    threading.Thread(
        target=plot_thread,
        daemon=True
    ).start()


# ============================================================
# PLOT THREAD
# ============================================================

def plot_thread():

    global plotting

    try:

        send_command(
            "HOME"
        )

        time.sleep(
            0.5
        )

        send_command(
            "PENUP"
        )

        for path_number, path in enumerate(
            toolpaths
        ):

            if not plotting:
                break

            if len(path) < 2:
                continue

            # ------------------------------------------------
            # MOVE TO START
            # ------------------------------------------------

            x, y = path[0]

            send_command(
                f"M X{mm_to_steps_x(x)} "
                f"Y{mm_to_steps_y(y)}"
            )

            # ------------------------------------------------
            # PEN DOWN
            # ------------------------------------------------

            send_command(
                "PENDOWN"
            )

            # ------------------------------------------------
            # DRAW
            # ------------------------------------------------

            for x, y in path[1:]:

                if not plotting:
                    break

                send_command(
                    f"M X{mm_to_steps_x(x)} "
                    f"Y{mm_to_steps_y(y)}"
                )

            # ------------------------------------------------
            # PEN UP
            # ------------------------------------------------

            send_command(
                "PENUP"
            )

            status_var.set(
                f"Plotting path "
                f"{path_number + 1}/"
                f"{len(toolpaths)}"
            )

        if plotting:

            send_command(
                "PENUP"
            )

            send_command(
                "HOME"
            )

            status_var.set(
                "Plot complete"
            )

    except Exception as e:

        status_var.set(
            "Plot error: " +
            str(e)
        )

    plotting = False

    plot_button.config(
        state="normal"
    )


# ============================================================
# STOP
# ============================================================

def stop_plot():

    global plotting

    plotting = False

    if ser:

        try:

            send_command(
                "PENUP"
            )

            send_command(
                "HOME"
            )

        except Exception:
            pass

    status_var.set(
        "Plot stopped"
    )


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Arduino 2D Plotter - Vectorizer"
)

root.geometry(
    "1200x750"
)

root.minsize(
    950,
    600
)


# ============================================================
# MAIN
# ============================================================

main = tk.Frame(
    root
)

main.pack(
    fill="both",
    expand=True
)


# ============================================================
# LEFT SCROLLABLE PANEL
# ============================================================

left_container = tk.Frame(
    main,
    width=350
)

left_container.pack(
    side="left",
    fill="y"
)

left_container.pack_propagate(
    False
)


left_canvas = tk.Canvas(
    left_container,
    highlightthickness=0
)

left_scrollbar = ttk.Scrollbar(
    left_container,
    orient="vertical",
    command=left_canvas.yview
)

left_canvas.configure(
    yscrollcommand=
    left_scrollbar.set
)

left_scrollbar.pack(
    side="right",
    fill="y"
)

left_canvas.pack(
    side="left",
    fill="both",
    expand=True
)


left = tk.Frame(
    left_canvas
)

left_window = left_canvas.create_window(
    (0, 0),
    window=left,
    anchor="nw"
)


def update_scroll_region(
    event=None
):

    left_canvas.configure(
        scrollregion=
        left_canvas.bbox("all")
    )


def resize_left_frame(
    event
):

    left_canvas.itemconfig(
        left_window,
        width=event.width
    )


left.bind(
    "<Configure>",
    update_scroll_region
)

left_canvas.bind(
    "<Configure>",
    resize_left_frame
)


def mousewheel(event):

    left_canvas.yview_scroll(
        int(
            -1 *
            (event.delta / 120)
        ),
        "units"
    )


left_canvas.bind_all(
    "<MouseWheel>",
    mousewheel
)


# ============================================================
# BASIC SETTINGS
# ============================================================

basic = CollapsibleFrame(
    left,
    "Basic Settings",
    expanded=True
)

basic.pack(
    fill="x",
    padx=8,
    pady=5
)


tk.Button(
    basic.content,
    text="OPEN IMAGE",
    command=open_image,
    width=30
).pack(
    pady=5
)


# ------------------------------------------------------------
# VECTORIZATION METHOD
# ------------------------------------------------------------

tk.Label(
    basic.content,
    text="Vectorization Method",
    font=("Arial", 9, "bold")
).pack(
    pady=(8, 2)
)


vectorization_mode = ttk.Combobox(
    basic.content,
    values=[
        "SVG Trace - Many Paths",
        "Meaningful Strokes / Edges"
    ],
    state="readonly",
    width=28
)

vectorization_mode.current(
    0
)

vectorization_mode.pack(
    pady=3
)


tk.Label(
    basic.content,
    text=(
        "Many Paths = regions/outlines\n"
        "Strokes = photographic line details"
    ),
    justify="left"
).pack(
    anchor="w",
    pady=(2, 6)
)


# ------------------------------------------------------------
# PROCESSING
# ------------------------------------------------------------

tk.Label(
    basic.content,
    text="Pre-processing Mode",
    font=("Arial", 9, "bold")
).pack(
    pady=(4, 2)
)


processing_mode = ttk.Combobox(
    basic.content,
    values=[
        "Threshold",
        "Edge Detection",
        "Adaptive"
    ],
    state="readonly",
    width=28
)

processing_mode.current(
    0
)

processing_mode.pack(
    pady=3
)

processing_mode.bind(
    "<<ComboboxSelected>>",
    lambda e: process_image()
)


tk.Label(
    basic.content,
    text="Threshold"
).pack(
    pady=(5, 0)
)


threshold_var = tk.IntVar(
    value=DEFAULT_THRESHOLD
)


tk.Scale(
    basic.content,
    from_=0,
    to=255,
    orient="horizontal",
    variable=threshold_var,
    command=lambda x: process_image(),
    length=240
).pack()


# ------------------------------------------------------------
# PLOT SIZE
# ------------------------------------------------------------

tk.Label(
    basic.content,
    text="Plot Width (mm)"
).pack(
    pady=(8, 2)
)


width_var = tk.StringVar(
    value=str(
        DEFAULT_WIDTH
    )
)


tk.Entry(
    basic.content,
    textvariable=width_var,
    width=30
).pack()


tk.Label(
    basic.content,
    text="Plot Height (mm)"
).pack(
    pady=(5, 2)
)


height_var = tk.StringVar(
    value=str(
        DEFAULT_HEIGHT
    )
)


tk.Entry(
    basic.content,
    textvariable=height_var,
    width=30
).pack()


# ============================================================
# VECTORIZATION SETTINGS
# ============================================================

vector_settings = CollapsibleFrame(
    left,
    "Vectorization Settings",
    expanded=True
)

vector_settings.pack(
    fill="x",
    padx=8,
    pady=5
)


# ------------------------------------------------------------
# Edge low
# ------------------------------------------------------------

tk.Label(
    vector_settings.content,
    text="Edge Low Threshold"
).pack(
    pady=(2, 2)
)


edge_low_var = tk.StringVar(
    value=str(
        DEFAULT_EDGE_LOW
    )
)


tk.Entry(
    vector_settings.content,
    textvariable=edge_low_var,
    width=30
).pack()


# ------------------------------------------------------------
# Edge high
# ------------------------------------------------------------

tk.Label(
    vector_settings.content,
    text="Edge High Threshold"
).pack(
    pady=(6, 2)
)


edge_high_var = tk.StringVar(
    value=str(
        DEFAULT_EDGE_HIGH
    )
)


tk.Entry(
    vector_settings.content,
    textvariable=edge_high_var,
    width=30
).pack()


# ------------------------------------------------------------
# Blur
# ------------------------------------------------------------

tk.Label(
    vector_settings.content,
    text="Blur Kernel Size"
).pack(
    pady=(6, 2)
)


blur_var = tk.StringVar(
    value=str(
        DEFAULT_BLUR_SIZE
    )
)


tk.Entry(
    vector_settings.content,
    textvariable=blur_var,
    width=30
).pack()


tk.Label(
    vector_settings.content,
    text="Odd number: 1, 3, 5, 7..."
).pack(
    anchor="w"
)


# ------------------------------------------------------------
# Adaptive block
# ------------------------------------------------------------

tk.Label(
    vector_settings.content,
    text="Adaptive Block Size"
).pack(
    pady=(6, 2)
)


adaptive_block_var = tk.StringVar(
    value="11"
)


tk.Entry(
    vector_settings.content,
    textvariable=adaptive_block_var,
    width=30
).pack()


# ------------------------------------------------------------
# Adaptive C
# ------------------------------------------------------------

tk.Label(
    vector_settings.content,
    text="Adaptive C"
).pack(
    pady=(6, 2)
)


adaptive_c_var = tk.StringVar(
    value="2"
)


tk.Entry(
    vector_settings.content,
    textvariable=adaptive_c_var,
    width=30
).pack()


# ============================================================
# ADVANCED PATH SETTINGS
# ============================================================

advanced = CollapsibleFrame(
    left,
    "Advanced Path Settings",
    expanded=False
)

advanced.pack(
    fill="x",
    padx=8,
    pady=5
)


# ------------------------------------------------------------
# Simplify
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Simplify Epsilon (mm)"
).pack(
    pady=(2, 2)
)


simplify_var = tk.StringVar(
    value=str(
        DEFAULT_SIMPLIFY_EPSILON
    )
)


tk.Entry(
    advanced.content,
    textvariable=simplify_var,
    width=30
).pack()


tk.Label(
    advanced.content,
    text="Higher = fewer vector points"
).pack(
    anchor="w"
)


# ------------------------------------------------------------
# Min contour length
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Min Contour Length (px)"
).pack(
    pady=(7, 2)
)


min_contour_length_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_CONTOUR_LENGTH
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_contour_length_var,
    width=30
).pack()


# ------------------------------------------------------------
# Min contour area
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Min Contour Area (px²)"
).pack(
    pady=(7, 2)
)


min_contour_area_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_CONTOUR_AREA
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_contour_area_var,
    width=30
).pack()


# ------------------------------------------------------------
# Min path
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Min Path Length (mm)"
).pack(
    pady=(7, 2)
)


min_path_length_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_PATH_LENGTH
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_path_length_var,
    width=30
).pack()


# ------------------------------------------------------------
# Min stroke
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Min Stroke Length (mm)"
).pack(
    pady=(7, 2)
)


min_stroke_length_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_STROKE_LENGTH
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_stroke_length_var,
    width=30
).pack()


# ------------------------------------------------------------
# Min point distance
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Min Point Distance (mm)"
).pack(
    pady=(7, 2)
)


min_point_distance_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_POINT_DISTANCE
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_point_distance_var,
    width=30
).pack()


# ------------------------------------------------------------
# Morphological kernel
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Morphological Kernel"
).pack(
    pady=(7, 2)
)


morph_kernel_var = tk.StringVar(
    value=str(
        DEFAULT_MORPH_KERNEL_SIZE
    )
)


tk.Entry(
    advanced.content,
    textvariable=morph_kernel_var,
    width=30
).pack()


tk.Label(
    advanced.content,
    text="Odd numbers recommended"
).pack(
    anchor="w"
)


# ------------------------------------------------------------
# Maximum paths
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Maximum Paths"
).pack(
    pady=(7, 2)
)


max_paths_var = tk.StringVar(
    value=str(
        DEFAULT_MAX_PATHS
    )
)


tk.Entry(
    advanced.content,
    textvariable=max_paths_var,
    width=30
).pack()


# ------------------------------------------------------------
# Duplicate tolerance
# ------------------------------------------------------------

tk.Label(
    advanced.content,
    text="Duplicate Tolerance (mm)"
).pack(
    pady=(7, 2)
)


duplicate_tolerance_var = tk.StringVar(
    value=str(
        DEFAULT_DUPLICATE_TOLERANCE
    )
)


tk.Entry(
    advanced.content,
    textvariable=duplicate_tolerance_var,
    width=30
).pack()


# ------------------------------------------------------------
# Closed contours
# ------------------------------------------------------------

closed_contours_var = tk.BooleanVar(
    value=True
)


tk.Checkbutton(
    advanced.content,
    text="Close traced contours",
    variable=closed_contours_var
).pack(
    anchor="w",
    pady=(8, 2)
)


# ------------------------------------------------------------
# Optimize
# ------------------------------------------------------------

optimize_paths_var = tk.BooleanVar(
    value=True
)


tk.Checkbutton(
    advanced.content,
    text="Optimize path order",
    variable=optimize_paths_var
).pack(
    anchor="w",
    pady=2
)


# ============================================================
# PLOTTER SETTINGS
# ============================================================

plotter_settings = CollapsibleFrame(
    left,
    "Plotter Settings",
    expanded=False
)

plotter_settings.pack(
    fill="x",
    padx=8,
    pady=5
)


# ------------------------------------------------------------
# X
# ------------------------------------------------------------

tk.Label(
    plotter_settings.content,
    text="X Steps/mm"
).pack(
    pady=(2, 2)
)


x_steps_var = tk.StringVar(
    value=str(
        DEFAULT_X_STEPS_MM
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=x_steps_var,
    width=30
).pack()


# ------------------------------------------------------------
# Y
# ------------------------------------------------------------

tk.Label(
    plotter_settings.content,
    text="Y Steps/mm"
).pack(
    pady=(7, 2)
)


y_steps_var = tk.StringVar(
    value=str(
        DEFAULT_Y_STEPS_MM
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=y_steps_var,
    width=30
).pack()


# ------------------------------------------------------------
# Pen up
# ------------------------------------------------------------

tk.Label(
    plotter_settings.content,
    text="Pen Up Angle"
).pack(
    pady=(7, 2)
)


pen_up_var = tk.StringVar(
    value=str(
        DEFAULT_PEN_UP
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=pen_up_var,
    width=30
).pack()


# ------------------------------------------------------------
# Pen down
# ------------------------------------------------------------

tk.Label(
    plotter_settings.content,
    text="Pen Down Angle"
).pack(
    pady=(7, 2)
)


pen_down_var = tk.StringVar(
    value=str(
        DEFAULT_PEN_DOWN
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=pen_down_var,
    width=30
).pack()


# ============================================================
# SERIAL SETTINGS
# ============================================================

serial_settings = CollapsibleFrame(
    left,
    "Serial Settings",
    expanded=False
)

serial_settings.pack(
    fill="x",
    padx=8,
    pady=5
)


tk.Label(
    serial_settings.content,
    text="Arduino Port"
).pack(
    pady=(2, 2)
)


port_var = tk.StringVar()


port_combo = ttk.Combobox(
    serial_settings.content,
    textvariable=port_var,
    width=27
)

port_combo.pack(
    pady=3
)


tk.Button(
    serial_settings.content,
    text="Refresh Ports",
    command=refresh_ports,
    width=30
).pack(
    pady=3
)


connect_button = tk.Button(
    serial_settings.content,
    text="Connect",
    command=connect_serial,
    width=30
)

connect_button.pack(
    pady=3
)


# ============================================================
# ACTIONS
# ============================================================

actions = CollapsibleFrame(
    left,
    "Actions",
    expanded=True
)

actions.pack(
    fill="x",
    padx=8,
    pady=5
)


tk.Button(
    actions.content,
    text="GENERATE VECTOR TOOLPATH",
    command=generate_toolpaths,
    width=30,
    height=2
).pack(
    pady=4
)


tk.Button(
    actions.content,
    text="EXPORT SVG",
    command=export_svg,
    width=30
).pack(
    pady=3
)


plot_button = tk.Button(
    actions.content,
    text="PLOT",
    command=start_plot,
    width=30,
    height=2
)

plot_button.pack(
    pady=4
)


tk.Button(
    actions.content,
    text="STOP",
    command=stop_plot,
    width=30,
    height=2
).pack(
    pady=4
)


# ============================================================
# RIGHT PANEL
# ============================================================

right = tk.Frame(
    main
)

right.pack(
    side="right",
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


tk.Label(
    right,
    text="Image / Vector Toolpath Preview",
    font=("Arial", 16)
).pack(
    pady=5
)


image_label = tk.Label(
    right
)

image_label.pack(
    pady=5
)


canvas = tk.Canvas(
    right,
    width=650,
    height=500,
    bg="white"
)

canvas.pack(
    pady=5
)


# ============================================================
# STATUS
# ============================================================

status_var = tk.StringVar(
    value="Ready"
)


tk.Label(
    right,
    textvariable=status_var
).pack(
    pady=5
)


# ============================================================
# START
# ============================================================

refresh_ports()

root.mainloop()