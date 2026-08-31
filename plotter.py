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

        self.parent = parent
        self.expanded = expanded

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        self.header = tk.Frame(
            self
        )

        self.header.pack(
            fill="x"
        )

        self.toggle_button = tk.Button(
            self.header,
            text="▼ " + title,
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

        # ----------------------------------------------------
        # CONTENT
        # ----------------------------------------------------

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

        # Update scroll region
        left_canvas.update_idletasks()

        left_canvas.configure(
            scrollregion=
            left_canvas.bbox("all")
        )


# ============================================================
# IMAGE
# ============================================================

def open_image():

    global image_original

    filename = filedialog.askopenfilename(
        filetypes=[
            (
                "Image files",
                "*.png *.jpg *.jpeg *.bmp"
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

    show_image(
        img
    )

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
        (500, 300)
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

        x1, y1 = path[i - 1]
        x2, y2 = path[i]

        total += math.hypot(
            x2 - x1,
            y2 - y1
        )

    if closed and len(path) >= 3:

        x1, y1 = path[-1]
        x2, y2 = path[0]

        total += math.hypot(
            x2 - x1,
            y2 - y1
        )

    return total


# ============================================================
# REMOVE DUPLICATE PATHS
# ============================================================

def remove_duplicate_paths(paths):

    if not paths:
        return []

    unique = []

    for path in paths:

        if len(path) < 3:
            continue

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

            e_min_x = min(ex)
            e_max_x = max(ex)

            e_min_y = min(ey)
            e_max_y = max(ey)

            e_width = e_max_x - e_min_x
            e_height = e_max_y - e_min_y

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

                duplicate = True

                break

        if not duplicate:

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
# GENERATE TOOLPATH
# ============================================================

def generate_toolpaths():

    global toolpaths

    if image_processed is None:

        messagebox.showwarning(
            "Warning",
            "Open an image first."
        )

        return

    # --------------------------------------------------------
    # READ SETTINGS
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

    except ValueError:

        messagebox.showerror(
            "Error",
            "Invalid vector settings."
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

    # --------------------------------------------------------
    # FORCE ODD KERNEL
    # --------------------------------------------------------

    if morph_kernel_size < 1:

        morph_kernel_size = 1

    if morph_kernel_size % 2 == 0:

        morph_kernel_size += 1

    # --------------------------------------------------------
    # BINARY
    # --------------------------------------------------------

    img = image_processed.copy()

    mode = processing_mode.get()

    if mode == "Edge Detection":

        binary = img.copy()

    else:

        _, binary = cv2.threshold(
            img,
            127,
            255,
            cv2.THRESH_BINARY_INV
        )

    # --------------------------------------------------------
    # MORPHOLOGICAL CLEANUP
    # --------------------------------------------------------

    if morph_kernel_size > 1:

        kernel = np.ones(
            (
                morph_kernel_size,
                morph_kernel_size
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

    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None:

        toolpaths = []

        draw_toolpath_preview()

        status_var.set(
            "No contours found"
        )

        return

    h, w = binary.shape

    scale_x = width_mm / w
    scale_y = height_mm / h

    paths = []

    # --------------------------------------------------------
    # CONTOURS
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
            True
        )

        if perimeter < min_contour_length:
            continue

        # ----------------------------------------------------
        # SIMPLIFY
        # ----------------------------------------------------

        epsilon = (
            simplify_epsilon
            * scale_x
        )

        simplified = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(simplified) < 3:
            continue

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
            path,
            min_point_distance
        )

        if len(path) < 3:
            continue

        # ----------------------------------------------------
        # PATH LENGTH
        # ----------------------------------------------------

        length = calculate_path_length(
            path,
            closed=True
        )

        if length < min_path_length:
            continue

        paths.append(
            path
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    paths = remove_duplicate_paths(
        paths
    )

    # --------------------------------------------------------
    # OPTIMIZE
    # --------------------------------------------------------

    toolpaths = optimize_paths(
        paths
    )

    draw_toolpath_preview()

    status_var.set(
        f"{len(toolpaths)} paths generated"
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
        height * scale
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
                dash=(3, 3)
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

        # Close contour

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
        # DRAW PATHS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

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
    "1150x700"
)

root.minsize(
    900,
    600
)


# ============================================================
# MAIN LAYOUT
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
    width=330
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
    (
        0,
        0
    ),
    window=left,
    anchor="nw"
)


def update_scroll_region(event=None):

    left_canvas.configure(
        scrollregion=
        left_canvas.bbox("all")
    )


def resize_left_frame(event):

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


# ============================================================
# MOUSE WHEEL SCROLL
# ============================================================

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
    pady=6
)


tk.Button(
    basic.content,
    text="OPEN IMAGE",
    command=open_image,
    width=28
).pack(
    pady=5
)


tk.Label(
    basic.content,
    text="Processing Mode"
).pack(
    pady=(8, 2)
)


processing_mode = ttk.Combobox(
    basic.content,
    values=[
        "Threshold",
        "Edge Detection",
        "Adaptive"
    ],
    state="readonly",
    width=25
)

processing_mode.current(
    0
)

processing_mode.pack(
    pady=3
)

processing_mode.bind(
    "<<ComboboxSelected>>",
    lambda e:
    process_image()
)


tk.Label(
    basic.content,
    text="Threshold"
).pack(
    pady=(6, 0)
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
    command=lambda x:
    process_image(),
    length=220
).pack()


# ============================================================
# PLOT SIZE
# ============================================================

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
    width=28
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
    width=28
).pack()


# ============================================================
# ADVANCED SETTINGS
# ============================================================

advanced = CollapsibleFrame(
    left,
    "Advanced Settings",
    expanded=False
)

advanced.pack(
    fill="x",
    padx=8,
    pady=6
)


# Simplify

tk.Label(
    advanced.content,
    text="Simplify Epsilon"
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
    width=28
).pack()


tk.Label(
    advanced.content,
    text="Higher = fewer points"
).pack(
    anchor="w"
)


# Minimum contour length

tk.Label(
    advanced.content,
    text="Min Contour Length"
).pack(
    pady=(8, 2)
)


min_contour_length_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_CONTOUR_LENGTH
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_contour_length_var,
    width=28
).pack()


# Minimum contour area

tk.Label(
    advanced.content,
    text="Min Contour Area"
).pack(
    pady=(8, 2)
)


min_contour_area_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_CONTOUR_AREA
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_contour_area_var,
    width=28
).pack()


# Minimum path length

tk.Label(
    advanced.content,
    text="Min Path Length (mm)"
).pack(
    pady=(8, 2)
)


min_path_length_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_PATH_LENGTH
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_path_length_var,
    width=28
).pack()


# Minimum point distance

tk.Label(
    advanced.content,
    text="Min Point Distance (mm)"
).pack(
    pady=(8, 2)
)


min_point_distance_var = tk.StringVar(
    value=str(
        DEFAULT_MIN_POINT_DISTANCE
    )
)


tk.Entry(
    advanced.content,
    textvariable=min_point_distance_var,
    width=28
).pack()


# Morph kernel

tk.Label(
    advanced.content,
    text="Morphological Kernel"
).pack(
    pady=(8, 2)
)


morph_kernel_var = tk.StringVar(
    value=str(
        DEFAULT_MORPH_KERNEL_SIZE
    )
)


tk.Entry(
    advanced.content,
    textvariable=morph_kernel_var,
    width=28
).pack()


tk.Label(
    advanced.content,
    text="Odd numbers recommended"
).pack(
    anchor="w"
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
    pady=6
)


# X steps

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
    width=28
).pack()


# Y steps

tk.Label(
    plotter_settings.content,
    text="Y Steps/mm"
).pack(
    pady=(8, 2)
)


y_steps_var = tk.StringVar(
    value=str(
        DEFAULT_Y_STEPS_MM
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=y_steps_var,
    width=28
).pack()


# Pen up

tk.Label(
    plotter_settings.content,
    text="Pen Up Angle"
).pack(
    pady=(8, 2)
)


pen_up_var = tk.StringVar(
    value=str(
        DEFAULT_PEN_UP
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=pen_up_var,
    width=28
).pack()


# Pen down

tk.Label(
    plotter_settings.content,
    text="Pen Down Angle"
).pack(
    pady=(8, 2)
)


pen_down_var = tk.StringVar(
    value=str(
        DEFAULT_PEN_DOWN
    )
)


tk.Entry(
    plotter_settings.content,
    textvariable=pen_down_var,
    width=28
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
    pady=6
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
    width=25
)


port_combo.pack(
    pady=3
)


tk.Button(
    serial_settings.content,
    text="Refresh Ports",
    command=refresh_ports,
    width=28
).pack(
    pady=3
)


connect_button = tk.Button(
    serial_settings.content,
    text="Connect",
    command=connect_serial,
    width=28
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
    pady=6
)


tk.Button(
    actions.content,
    text="GENERATE TOOLPATH",
    command=generate_toolpaths,
    width=28,
    height=2
).pack(
    pady=4
)


plot_button = tk.Button(
    actions.content,
    text="PLOT",
    command=start_plot,
    width=28,
    height=2
)


plot_button.pack(
    pady=4
)


tk.Button(
    actions.content,
    text="STOP",
    command=stop_plot,
    width=28,
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
    text="Image / Toolpath Preview",
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