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

MIN_CONTOUR_LENGTH = 10
SIMPLIFY_EPSILON = 1.0


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

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur removes tiny image noise
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
    # EDGE
    # --------------------------------------------------------

    elif mode == "Edge Detection":

        edges = cv2.Canny(
            gray,
            threshold,
            min(255, threshold * 2)
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

    img = image_processed.copy()

    # Make sure contours see white objects
    if processing_mode.get() == "Edge Detection":

        binary = img

    else:

        _, binary = cv2.threshold(
            img,
            127,
            255,
            cv2.THRESH_BINARY_INV
        )

    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE
    )

    h, w = binary.shape

    width_mm = float(
        width_var.get()
    )

    height_mm = float(
        height_var.get()
    )

    scale_x = width_mm / w
    scale_y = height_mm / h

    paths = []

    # --------------------------------------------------------
    # CONVERT CONTOURS TO MM
    # --------------------------------------------------------

    for contour in contours:

        if len(contour) < 2:
            continue

        perimeter = cv2.arcLength(
            contour,
            False
        )

        if perimeter < MIN_CONTOUR_LENGTH:
            continue

        epsilon = (
            SIMPLIFY_EPSILON
            * perimeter
            / 100
        )

        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            False
        )

        path = []

        for p in simplified:

            x = float(
                p[0][0]
            )

            y = float(
                p[0][1]
            )

            # Image Y → plotter Y
            px = x * scale_x
            py = y * scale_y

            path.append(
                (px, py)
            )

        if len(path) >= 2:

            length = calculate_path_length(
                path
            )

            if length > 1:

                paths.append(
                    path
                )

    # --------------------------------------------------------
    # OPTIMIZE PATHS
    # --------------------------------------------------------

    toolpaths = optimize_paths(
        paths
    )

    draw_toolpath_preview()

    status_var.set(
        f"{len(toolpaths)} paths generated"
    )


# ============================================================
# PATH LENGTH
# ============================================================

def calculate_path_length(path):

    total = 0

    for i in range(1, len(path)):

        x1, y1 = path[i - 1]
        x2, y2 = path[i]

        total += math.hypot(
            x2 - x1,
            y2 - y1
        )

    return total


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return math.hypot(
        b[0] - a[0],
        b[1] - a[1]
    )


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

    remaining = paths.copy()

    optimized = []

    current = (0, 0)

    while remaining:

        best_index = 0
        best_distance = float("inf")
        best_reverse = False

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
# PREVIEW
# ============================================================

def draw_toolpath_preview():

    canvas.delete(
        "all"
    )

    if not toolpaths:
        return

    width = float(
        width_var.get()
    )

    height = float(
        height_var.get()
    )

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

    # Plot boundary

    canvas.create_rectangle(
        offset_x,
        offset_y,
        offset_x + width * scale,
        offset_y + height * scale
    )

    # Paths

    for path in toolpaths:

        if len(path) < 2:
            continue

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
                [sx, sy]
            )

        canvas.create_line(
            *points,
            width=1
        )


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

        ser = serial.Serial(
            port,
            BAUD_RATE,
            timeout=5
        )

        time.sleep(2)

        status_var.set(
            f"Connected to {port}"
        )

        connect_button.config(
            text="Connected"
        )

    except Exception as e:

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
            (command + "\n").encode()
        )

        response = ser.readline()

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
        "Make sure the pen is positioned at the HOME position.\n\nStart plotting?"
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


def plot_thread():

    global plotting

    try:

        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        send_command(
            "HOME"
        )

        time.sleep(
            0.5
        )

        # ----------------------------------------------------
        # PEN UP
        # ----------------------------------------------------

        send_command(
            "PENUP"
        )

        # ----------------------------------------------------
        # PATHS
        # ----------------------------------------------------

        for path_number, path in enumerate(
            toolpaths
        ):

            if not plotting:
                break

            if len(path) < 2:
                continue

            # Move to first point

            x, y = path[0]

            send_command(
                f"M X{mm_to_steps_x(x)} "
                f"Y{mm_to_steps_y(y)}"
            )

            # Pen down

            send_command(
                "PENDOWN"
            )

            # Draw

            for x, y in path:

                if not plotting:
                    break

                send_command(
                    f"M X{mm_to_steps_x(x)} "
                    f"Y{mm_to_steps_y(y)}"
                )

            # Pen up

            send_command(
                "PENUP"
            )

            status_var.set(
                f"Plotting path "
                f"{path_number + 1}/"
                f"{len(toolpaths)}"
            )

        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        if plotting:

            send_command(
                "HOME"
            )

            status_var.set(
                "Plot complete"
            )

    except Exception as e:

        status_var.set(
            "Plot error: " + str(e)
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

        send_command("PENUP")
        send_command("HOME")

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


tk.Button(
    left,
    text="OPEN IMAGE",
    command=open_image,
    width=25
).pack(
    pady=5
)


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


status_var = tk.StringVar(
    value="Ready"
)

tk.Label(
    right,
    textvariable=status_var
).pack()


refresh_ports()

root.mainloop()