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
# SETTINGS
# ============================================================

BAUD_RATE = 115200

DEFAULT_X_STEPS_MM = 21.28
DEFAULT_Y_STEPS_MM = 20.95

DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100

DEFAULT_PEN_UP = 90
DEFAULT_PEN_DOWN = 30

# Toolpath filtering
MIN_CONTOUR_LENGTH = 15
MIN_CONTOUR_AREA = 5
MIN_PATH_LENGTH = 1.5

# Contour simplification
SIMPLIFY_EPSILON = 0.8

# Image cleanup
MORPH_KERNEL_SIZE = 3

# Remove points that are extremely close
MIN_POINT_DISTANCE = 0.2


# ============================================================
# GLOBALS
# ============================================================

image_original = None
image_processed = None

toolpaths = []

ser = None
plotting = False


# ============================================================
# IMAGE
# ============================================================

def open_image():

    global image_original

    filename = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp"),
            ("All files", "*.*")
        ]
    )

    if not filename:
        return

    img = cv2.imread(filename)

    if img is None:

        messagebox.showerror(
            "Error",
            "Could not load image."
        )

        return

    image_original = img

    show_image(img)

    process_image()

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

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur removes tiny pixel noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    mode = processing_mode.get()

    threshold = threshold_var.get()

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    if mode == "Threshold":

        _, processed = cv2.threshold(
            gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

    # --------------------------------------------------------
    # EDGE DETECTION
    # --------------------------------------------------------

    elif mode == "Edge Detection":

        edges = cv2.Canny(
            gray,
            threshold,
            min(
                255,
                threshold * 2
            )
        )

        processed = edges

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    elif mode == "Adaptive":

        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    else:

        processed = gray

    image_processed = processed

    show_image(processed)


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

    pil = Image.fromarray(display)

    pil.thumbnail(
        (450, 350)
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

def remove_close_points(path):

    if len(path) < 2:
        return path

    cleaned = [
        path[0]
    ]

    for point in path[1:]:

        if distance(
            cleaned[-1],
            point
        ) >= MIN_POINT_DISTANCE:

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

        x1, y1 = path[i - 1]
        x2, y2 = path[i]

        total += math.hypot(
            x2 - x1,
            y2 - y1
        )

    # Close contour
    if closed and len(path) >= 3:

        x1, y1 = path[-1]
        x2, y2 = path[0]

        total += math.hypot(
            x2 - x1,
            y2 - y1
        )

    return total


# ============================================================
# CONTOUR EXTRACTION
# ============================================================

def generate_toolpaths():

    global toolpaths

    if image_processed is None:

        messagebox.showwarning(
            "Warning",
            "Open an image first."
        )

        return

    # ========================================================
    # VALIDATE SIZE
    # ========================================================

    try:

        width_mm = float(
            width_var.get()
        )

        height_mm = float(
            height_var.get()
        )

    except ValueError:

        messagebox.showerror(
            "Error",
            "Invalid plot width or height."
        )

        return

    if width_mm <= 0 or height_mm <= 0:

        messagebox.showerror(
            "Error",
            "Plot width and height must be greater than zero."
        )

        return

    # ========================================================
    # COPY IMAGE
    # ========================================================

    img = image_processed.copy()

    mode = processing_mode.get()

    # ========================================================
    # CREATE BINARY IMAGE
    # ========================================================

    if mode == "Edge Detection":

        # Canny already gives a binary image
        binary = img.copy()

        # Close small gaps in edge lines
        kernel = np.ones(
            (
                MORPH_KERNEL_SIZE,
                MORPH_KERNEL_SIZE
            ),
            np.uint8
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

    else:

        # Convert black areas to white
        _, binary = cv2.threshold(
            img,
            127,
            255,
            cv2.THRESH_BINARY_INV
        )

        kernel = np.ones(
            (
                MORPH_KERNEL_SIZE,
                MORPH_KERNEL_SIZE
            ),
            np.uint8
        )

        # Remove tiny isolated noise
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel
        )

        # Connect small gaps
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:

        toolpaths = []

        draw_toolpath_preview()

        status_var.set(
            "No contours found"
        )

        return

    hierarchy = hierarchy[0]

    h, w = binary.shape

    # ========================================================
    # PIXEL → MM SCALE
    # ========================================================

    scale_x = width_mm / w
    scale_y = height_mm / h

    paths = []

    # ========================================================
    # PROCESS CONTOURS
    # ========================================================

    for i, contour in enumerate(contours):

        if len(contour) < 3:
            continue

        # ----------------------------------------------------
        # AREA FILTER
        # ----------------------------------------------------

        area = abs(
            cv2.contourArea(
                contour
            )
        )

        if area < MIN_CONTOUR_AREA:
            continue

        # ----------------------------------------------------
        # PERIMETER
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter < MIN_CONTOUR_LENGTH:
            continue

        # ----------------------------------------------------
        # SIMPLIFY
        # ----------------------------------------------------

        simplified = cv2.approxPolyDP(
            contour,
            SIMPLIFY_EPSILON,
            True
        )

        if len(simplified) < 3:
            continue

        # ----------------------------------------------------
        # CONVERT TO MM
        # ----------------------------------------------------

        path = []

        for p in simplified:

            x = float(
                p[0][0]
            )

            y = float(
                p[0][1]
            )

            px = x * scale_x
            py = y * scale_y

            path.append(
                (
                    px,
                    py
                )
            )

        # ----------------------------------------------------
        # REMOVE CLOSE POINTS
        # ----------------------------------------------------

        path = remove_close_points(
            path
        )

        if len(path) < 3:
            continue

        # ----------------------------------------------------
        # LENGTH FILTER
        # ----------------------------------------------------

        length = calculate_path_length(
            path,
            closed=True
        )

        if length < MIN_PATH_LENGTH:
            continue

        paths.append(
            path
        )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    paths = remove_duplicate_paths(
        paths
    )

    # ========================================================
    # OPTIMIZE PATH ORDER
    # ========================================================

    toolpaths = optimize_paths(
        paths
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    draw_toolpath_preview()

    status_var.set(
        f"{len(toolpaths)} paths generated"
    )


# ============================================================
# REMOVE DUPLICATE PATHS
# ============================================================

def remove_duplicate_paths(paths):

    if not paths:
        return []

    unique = []

    for path in paths:

        is_duplicate = False

        xs = [
            p[0]
            for p in path
        ]

        ys = [
            p[1]
            for p in path
        ]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        for existing in unique:

            ex = [
                p[0]
                for p in existing
            ]

            ey = [
                p[1]
                for p in existing
            ]

            e_min_x = min(ex)
            e_max_x = max(ex)

            e_min_y = min(ey)
            e_max_y = max(ey)

            e_width = e_max_x - e_min_x
            e_height = e_max_y - e_min_y

            # ------------------------------------------------
            # Compare bounding boxes
            # ------------------------------------------------

            if (
                abs(
                    min_x -
                    e_min_x
                ) < 0.5
                and
                abs(
                    max_x -
                    e_max_x
                ) < 0.5
                and
                abs(
                    min_y -
                    e_min_y
                ) < 0.5
                and
                abs(
                    max_y -
                    e_max_y
                ) < 0.5
                and
                abs(
                    width -
                    e_width
                ) < 0.5
                and
                abs(
                    height -
                    e_height
                ) < 0.5
            ):

                is_duplicate = True

                break

        if not is_duplicate:

            unique.append(
                path
            )

    return unique


# ============================================================
# REVERSE PATH
# ============================================================

def reverse_path(path):

    return list(
        reversed(path)
    )


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

        best_distance = float(
            "inf"
        )

        best_reverse = False

        # ----------------------------------------------------
        # FIND CLOSEST PATH
        # ----------------------------------------------------

        for i, path in enumerate(
            remaining
        ):

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

        # ----------------------------------------------------
        # REMOVE PATH
        # ----------------------------------------------------

        path = remaining.pop(
            best_index
        )

        # ----------------------------------------------------
        # REVERSE IF CLOSER
        # ----------------------------------------------------

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

    canvas_width = 500
    canvas_height = 400

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

    # ========================================================
    # PLOT BOUNDARY
    # ========================================================

    canvas.create_rectangle(
        offset_x,
        offset_y,
        offset_x + width * scale,
        offset_y + height * scale
    )

    # ========================================================
    # DRAW PATHS
    # ========================================================

    previous_end = None

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
                previous_end[0] * scale
            )

            y1 = (
                offset_y +
                previous_end[1] * scale
            )

            x2 = (
                offset_x +
                start[0] * scale
            )

            y2 = (
                offset_y +
                start[1] * scale
            )

            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                dash=(3, 3),
                width=1
            )

        # ----------------------------------------------------
        # DRAW CONTOUR
        # ----------------------------------------------------

        points = []

        for x, y in path:

            sx = (
                offset_x +
                x * scale
            )

            sy = (
                offset_y +
                y * scale
            )

            points.extend(
                [
                    sx,
                    sy
                ]
            )

        # Close contour in preview
        x0, y0 = path[0]

        points.extend(
            [
                offset_x +
                x0 * scale,

                offset_y +
                y0 * scale
            ]
        )

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

        port_combo.current(0)


# ============================================================
# CONNECT SERIAL
# ============================================================

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

        # Arduino reset delay
        time.sleep(2)

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

        # Wait for Arduino response
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
# PLOT
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
        "Make sure the pen is positioned at the HOME position.\n\n"
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

        # ====================================================
        # HOME
        # ====================================================

        send_command(
            "HOME"
        )

        time.sleep(
            0.5
        )

        # ====================================================
        # PEN UP
        # ====================================================

        send_command(
            "PENUP"
        )

        # ====================================================
        # PATHS
        # ====================================================

        for path_number, path in enumerate(
            toolpaths
        ):

            if not plotting:
                break

            if len(path) < 2:
                continue

            # ------------------------------------------------
            # MOVE TO FIRST POINT
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
            # Start at second point because the first point
            # has already been reached.
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

        # ====================================================
        # RETURN HOME
        # ====================================================

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
    "Arduino 2D Plotter"
)

root.geometry(
    "1050x650"
)


# ============================================================
# LEFT PANEL
# ============================================================

left = tk.Frame(
    root
)

left.pack(
    side="left",
    fill="y",
    padx=10,
    pady=10
)


# ============================================================
# OPEN IMAGE
# ============================================================

tk.Button(
    left,
    text="OPEN IMAGE",
    command=open_image,
    width=25
).pack(
    pady=5
)


# ============================================================
# PROCESSING
# ============================================================

tk.Label(
    left,
    text="Processing"
).pack(
    pady=(10, 0)
)


processing_mode = ttk.Combobox(
    left,
    values=[
        "Threshold",
        "Edge Detection",
        "Adaptive"
    ],
    state="readonly",
    width=22
)

processing_mode.current(
    0
)

processing_mode.pack(
    pady=5
)

processing_mode.bind(
    "<<ComboboxSelected>>",
    lambda e: process_image()
)


# ============================================================
# THRESHOLD
# ============================================================

tk.Label(
    left,
    text="Threshold"
).pack()


threshold_var = tk.IntVar(
    value=150
)

tk.Scale(
    left,
    from_=0,
    to=255,
    orient="horizontal",
    variable=threshold_var,
    command=lambda x: process_image()
).pack()


# ============================================================
# SIZE
# ============================================================

tk.Label(
    left,
    text="Plot Width (mm)"
).pack(
    pady=(15, 0)
)

width_var = tk.StringVar(
    value=str(DEFAULT_WIDTH)
)

tk.Entry(
    left,
    textvariable=width_var,
    width=25
).pack()


tk.Label(
    left,
    text="Plot Height (mm)"
).pack()

height_var = tk.StringVar(
    value=str(DEFAULT_HEIGHT)
)

tk.Entry(
    left,
    textvariable=height_var,
    width=25
).pack()


# ============================================================
# CALIBRATION
# ============================================================

tk.Label(
    left,
    text="X Steps/mm"
).pack(
    pady=(15, 0)
)

x_steps_var = tk.StringVar(
    value=str(DEFAULT_X_STEPS_MM)
)

tk.Entry(
    left,
    textvariable=x_steps_var,
    width=25
).pack()


tk.Label(
    left,
    text="Y Steps/mm"
).pack()

y_steps_var = tk.StringVar(
    value=str(DEFAULT_Y_STEPS_MM)
)

tk.Entry(
    left,
    textvariable=y_steps_var,
    width=25
).pack()


# ============================================================
# PEN
# ============================================================

tk.Label(
    left,
    text="Pen Up Angle"
).pack(
    pady=(15, 0)
)

pen_up_var = tk.StringVar(
    value=str(DEFAULT_PEN_UP)
)

tk.Entry(
    left,
    textvariable=pen_up_var,
    width=25
).pack()


tk.Label(
    left,
    text="Pen Down Angle"
).pack()

pen_down_var = tk.StringVar(
    value=str(DEFAULT_PEN_DOWN)
)

tk.Entry(
    left,
    textvariable=pen_down_var,
    width=25
).pack()


# ============================================================
# SERIAL
# ============================================================

tk.Label(
    left,
    text="Arduino Port"
).pack(
    pady=(15, 0)
)

port_var = tk.StringVar()

port_combo = ttk.Combobox(
    left,
    textvariable=port_var,
    width=22
)

port_combo.pack()


tk.Button(
    left,
    text="Refresh Ports",
    command=refresh_ports
).pack(
    pady=3
)


connect_button = tk.Button(
    left,
    text="Connect",
    command=connect_serial,
    width=25
)

connect_button.pack(
    pady=3
)


# ============================================================
# ACTIONS
# ============================================================

tk.Button(
    left,
    text="GENERATE TOOLPATH",
    command=generate_toolpaths,
    width=25
).pack(
    pady=(15, 3)
)


plot_button = tk.Button(
    left,
    text="PLOT",
    command=start_plot,
    width=25
)

plot_button.pack(
    pady=3
)


tk.Button(
    left,
    text="STOP",
    command=stop_plot,
    width=25
).pack(
    pady=3
)


# ============================================================
# RIGHT PANEL
# ============================================================

right = tk.Frame(
    root
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
    text="Image / Toolpath Preview",
    font=("Arial", 16)
).pack()


image_label = tk.Label(
    right
)

image_label.pack(
    pady=5
)


canvas = tk.Canvas(
    right,
    width=500,
    height=400,
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
).pack()


# ============================================================
# START
# ============================================================

refresh_ports()

root.mainloop()