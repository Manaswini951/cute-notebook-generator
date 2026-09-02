import io
import os
import random
import math
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
    ImageOps
)

# Set page configuration for mobile devices
st.set_page_config(
    page_title="Custom Printable Notebook Generator",
    page_icon="📚",
    layout="centered"
)

# ============================================================
# PAGE SIZE CONFIGURATIONS (300 DPI)
# ============================================================

PAGE_SIZES = {
    "A4 Standard (8.27 x 11.69 in)": {"width": 2480, "height": 3508},
    "A5 Compact / Daily Journal (5.83 x 8.27 in)": {"width": 1748, "height": 2480},
    "US Letter (8.5 x 11 in)": {"width": 2550, "height": 3300}
}

DPI = 300
MAX_SIZE = 1800

STAR_COUNT = 10
SPARKLE_COUNT = 16

# ============================================================
# COLOR THEMES
# ============================================================

PASTEL_THEMES = [
    {
        "name": "Pink Dream",
        "background": (255, 239, 244),
        "accent": (244, 150, 180),
        "dark": (145, 70, 95),
        "line_color": (230, 190, 205),
        "shape1": (255, 194, 210),
        "shape2": (250, 220, 228)
    },
    {
        "name": "Sky Blue",
        "background": (235, 247, 255),
        "accent": (130, 190, 235),
        "dark": (65, 120, 165),
        "line_color": (190, 215, 235),
        "shape1": (185, 220, 245),
        "shape2": (215, 238, 250)
    },
    {
        "name": "Lavender",
        "background": (247, 239, 255),
        "accent": (190, 150, 235),
        "dark": (120, 80, 165),
        "line_color": (215, 195, 235),
        "shape1": (220, 195, 245),
        "shape2": (238, 225, 250)
    },
    {
        "name": "Mint Garden",
        "background": (235, 250, 242),
        "accent": (110, 195, 160),
        "dark": (55, 125, 100),
        "line_color": (185, 220, 205),
        "shape1": (175, 225, 205),
        "shape2": (210, 240, 225)
    },
    {
        "name": "Sunshine",
        "background": (255, 249, 225),
        "accent": (240, 190, 85),
        "dark": (155, 115, 35),
        "line_color": (235, 215, 165),
        "shape1": (255, 225, 145),
        "shape2": (250, 238, 195)
    }
]

DARK_THEMES = [
    {
        "name": "Midnight Galaxy",
        "background": (25, 28, 60),
        "accent": (255, 200, 90),
        "text": (255, 255, 255),
        "shape1": (80, 60, 150),
        "shape2": (35, 90, 160)
    },
    {
        "name": "Ocean Night",
        "background": (18, 50, 75),
        "accent": (255, 215, 110),
        "text": (255, 255, 255),
        "shape1": (30, 120, 150),
        "shape2": (60, 160, 180)
    },
    {
        "name": "Royal Purple",
        "background": (50, 25, 85),
        "accent": (255, 205, 110),
        "text": (255, 255, 255),
        "shape1": (120, 80, 180),
        "shape2": (170, 130, 220)
    },
    {
        "name": "Berry Pop",
        "background": (85, 20, 60),
        "accent": (255, 215, 110),
        "text": (255, 255, 255),
        "shape1": (180, 60, 115),
        "shape2": (220, 120, 160)
    }
]

# ============================================================
# DYNAMIC FONT HELPER
# ============================================================

def get_font(size, bold=False):
    font_names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    ]
    for font_path in font_names:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

# ============================================================
# ALGORITHMIC IMAGE PROCESSING
# ============================================================

def resize_image(img, max_size=MAX_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        return img.copy()
    scale = max_size / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)

def remove_small_components(mask, min_area=18):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned

def extract_clean_drawing_mask(img_rgb):
    arr = np.array(img_rgb).astype(np.uint8)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    bg_size = max(61, int(min(h, w) * 0.10))
    if bg_size % 2 == 0:
        bg_size += 1
    bg_gray = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
    local_darkness = bg_gray.astype(np.float32) - gray.astype(np.float32)

    sat_blur = cv2.GaussianBlur(sat, (35, 35), 0)
    color_difference = np.abs(sat - sat_blur)
    color_mask = np.where((sat > 38) | (color_difference > 14), 255, 0).astype(np.uint8)

    bh_size = max(21, int(min(h, w) * 0.035))
    if bh_size % 2 == 0:
        bh_size += 1
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bh_size, bh_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, bh_kernel)
    blackhat_mask = np.where(blackhat > 10, 255, 0).astype(np.uint8)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_mask = np.where(magnitude > 24.0, 255, 0).astype(np.uint8)

    relative_dark_mask = np.where(local_darkness > 13.0, 255, 0).astype(np.uint8)
    seeds = cv2.bitwise_or(relative_dark_mask, blackhat_mask)
    seeds = cv2.bitwise_or(seeds, color_mask)
    seeds = cv2.bitwise_or(seeds, edge_mask)

    very_dark = np.where(gray < 85, 255, 0).astype(np.uint8)
    structure_kernel_size = max(9, int(min(h, w) * 0.012))
    if structure_kernel_size % 2 == 0:
        structure_kernel_size += 1
    structure_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (structure_kernel_size, structure_kernel_size))
    nearby_structure = cv2.dilate(edge_mask, structure_kernel, iterations=1)
    protected_dark = cv2.bitwise_and(very_dark, nearby_structure)
    seeds = cv2.bitwise_or(seeds, protected_dark)

    close_size = max(5, int(min(h, w) * 0.008))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    mask = remove_small_components(mask, min_area=18)
    bw = max(6, int(min(h, w) * 0.004))
    mask[:bw, :] = 0
    mask[-bw:, :] = 0
    mask[:, :bw] = 0
    mask[:, -bw:] = 0

    return mask

def create_transparent_drawing(img):
    original_rgb = img.convert("RGB")
    mask = extract_clean_drawing_mask(original_rgb)
    mask = remove_small_components(mask, min_area=18)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask[mask < 18] = 0
    orig_arr = np.array(original_rgb)
    rgba_arr = np.dstack((orig_arr, mask))
    result = Image.fromarray(rgba_arr, "RGBA")

    bbox = result.getbbox()
    if bbox:
        result = result.crop(bbox)

    padding = 40
    padded = Image.new("RGBA", (result.width + padding * 2, result.height + padding * 2), (0, 0, 0, 0))
    padded.alpha_composite(result, (padding, padding))
    return padded

# ============================================================
# DECORATIONS & SHAPES
# ============================================================

def draw_star(draw, x, y, size, color):
    points = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        radius = size if i % 2 == 0 else size * 0.42
        px = x + math.cos(angle) * radius
        py = y + math.sin(angle) * radius
        points.append((px, py))
    draw.polygon(points, fill=color)

def draw_sparkle(draw, x, y, size, color):
    width = max(2, int(size * 0.18))
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)

def draw_bubble(draw, x, y, radius, color):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=4)
    draw.ellipse((x - radius * 0.5, y - radius * 0.6, x - radius * 0.2, y - radius * 0.3), fill=color)

def add_cover_decorations(page, theme, rng, page_w, page_h):
    draw = ImageDraw.Draw(page, "RGBA")
    accent = theme["accent"]

    for _ in range(STAR_COUNT + 10):
        x = rng.randint(80, page_w - 80)
        y = rng.randint(80, page_h - 80)
        draw_star(draw, x, y, rng.randint(15, 50), accent + (220,))

    for _ in range(SPARKLE_COUNT + 8):
        x = rng.randint(60, page_w - 60)
        y = rng.randint(60, page_h - 60)
        draw_sparkle(draw, x, y, rng.randint(12, 35), (255, 255, 255, 230))

    for _ in range(10):
        x = rng.randint(100, page_w - 100)
        y = rng.randint(100, page_h - 100)
        draw_bubble(draw, x, y, rng.randint(20, 55), accent + (180,))

    return page

def fit_drawing(drawing, max_width, max_height):
    w, h = drawing.size
    scale = min(max_width / w, max_height / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return drawing.resize(new_size, Image.Resampling.LANCZOS)

# ============================================================
# DYNAMIC COVER PAGE BUILDER
# ============================================================

def create_dynamic_cover_page(transparent_drawings, child_name, role_title, theme, seed, page_w, page_h):
    rng = random.Random(seed)
    cover = Image.new("RGBA", (page_w, page_h), theme["background"] + (255,))
    draw = ImageDraw.Draw(cover, "RGBA")

    # Dynamic Background Accents
    shape_style = rng.choice(["bubbles", "waves", "geometrics"])
    if shape_style == "bubbles":
        for _ in range(8):
            rx = rng.randint(-100, page_w + 100)
            ry = rng.randint(-100, page_h + 100)
            rad = rng.randint(int(page_w * 0.15), int(page_w * 0.35))
            col = rng.choice([theme["shape1"], theme["shape2"]])
            draw.ellipse((rx - rad, ry - rad, rx + rad, ry + rad), fill=col + (110,))
    elif shape_style == "waves":
        draw.ellipse((-int(page_w * 0.2), -int(page_h * 0.1), int(page_w * 0.6), int(page_h * 0.4)), fill=theme["shape1"] + (130,))
        draw.ellipse((int(page_w * 0.4), int(page_h * 0.6), int(page_w * 1.2), int(page_h * 1.1)), fill=theme["shape2"] + (130,))
    else:
        draw.polygon([(0, 0), (int(page_w * 0.7), 0), (0, int(page_h * 0.5))], fill=theme["shape1"] + (140,))
        draw.polygon([(page_w, page_h), (int(page_w * 0.3), page_h), (page_w, int(page_h * 0.5))], fill=theme["shape2"] + (140,))

    cover = add_cover_decorations(cover, theme, rng, page_w, page_h)
    draw = ImageDraw.Draw(cover, "RGBA")

    # Border
    border_inset = int(page_w * 0.035)
    draw.rounded_rectangle(
        (border_inset, border_inset, page_w - border_inset, page_h - border_inset),
        radius=40, outline=theme["accent"] + (220,), width=8
    )

    # Dynamic Title Sizing
    title_size = int(page_w * 0.045)
    subtitle_size = int(page_w * 0.024)

    title_font = get_font(title_size, bold=True)
    subtitle_font = get_font(subtitle_size, bold=False)

    notebook_title = "MY CREATIVE NOTEBOOK"
    bbox = draw.textbbox((0, 0), notebook_title, font=title_font)
    title_w = bbox[2] - bbox[0]
    title_y = int(page_h * 0.11)

    draw.text(((page_w - title_w) // 2 + 3, title_y + 3), notebook_title, font=title_font, fill=(0, 0, 0, 120))
    draw.text(((page_w - title_w) // 2, title_y), notebook_title, font=title_font, fill=theme["accent"])

    # Name & Role Badge Block
    if child_name or role_title:
        badge_w, badge_h = int(page_w * 0.65), int(page_h * 0.065)
        bx1 = (page_w - badge_w) // 2
        by1 = title_y + int(page_h * 0.05)
        draw.rounded_rectangle((bx1, by1, bx1 + badge_w, by1 + badge_h), radius=25, fill=(0, 0, 0, 140), outline=theme["accent"] + (180,), width=3)

        display_text = f"{child_name}" if child_name else "Notes & Ideas"
        if role_title:
            display_text += f" | {role_title}"

        bbox_n = draw.textbbox((0, 0), display_text, font=subtitle_font)
        name_w = bbox_n[2] - bbox_n[0]
        draw.text(((page_w - name_w) // 2, by1 + int(badge_h * 0.25)), display_text, font=subtitle_font, fill=theme["text"])

    # Center Drawing Compositions
    if transparent_drawings:
        max_item_w = int(page_w * 0.38)
        max_item_h = int(page_h * 0.32)

        positions = [
            (page_w // 2 - max_item_w // 2, page_h // 2 - max_item_h // 2 + 50),
            (page_w // 4 - 60, page_h // 2 - 180),
            (3 * page_w // 4 - max_item_w + 60, page_h // 2 - 180),
            (page_w // 4 - 60, page_h // 2 + 200),
            (3 * page_w // 4 - max_item_w + 60, page_h // 2 + 200)
        ]

        for i, img in enumerate(transparent_drawings[:5]):
            resized_img = fit_drawing(img, max_item_w, max_item_h)
            pos = positions[i if i < len(positions) else 0]
            cover.alpha_composite(resized_img, pos)

    return cover.convert("RGB")

# ============================================================
# INTERIOR LINED PAGE BUILDER
# ============================================================

def create_lined_notebook_page(transparent_drawing, theme, page_num, seed, page_w, page_h, schedule_slots=None):
    rng = random.Random(seed + page_num)
    page = Image.new("RGBA", (page_w, page_h), theme["background"] + (255,))

    draw = ImageDraw.Draw(page, "RGBA")
    draw.ellipse((-int(page_w * 0.15), -int(page_h * 0.1), int(page_w * 0.4), int(page_h * 0.25)), fill=theme["shape1"] + (120,))
    draw.ellipse((int(page_w * 0.7), int(page_h * 0.8), int(page_w * 1.2), int(page_h * 1.1)), fill=theme["shape2"] + (120,))

    # Watermark Background Drawing
    if transparent_drawing:
        faded_drawing = transparent_drawing.copy()
        max_wm_w, max_wm_h = int(page_w * 0.65), int(page_h * 0.45)
        scale = min(max_wm_w / faded_drawing.width, max_wm_h / faded_drawing.height)
        new_wm_size = (int(faded_drawing.width * scale), int(faded_drawing.height * scale))
        faded_drawing = faded_drawing.resize(new_wm_size, Image.Resampling.LANCZOS)

        alpha = faded_drawing.getchannel("A")
        alpha = alpha.point(lambda p: int(p * 0.20))
        faded_drawing.putalpha(alpha)

        wm_x = (page_w - faded_drawing.width) // 2
        wm_y = (page_h - faded_drawing.height) // 2
        page.alpha_composite(faded_drawing, (wm_x, wm_y))

    draw = ImageDraw.Draw(page, "RGBA")

    # Header Sizing
    header_font_size = int(page_w * 0.016)
    slot_font_size = int(page_w * 0.012)
    header_font = get_font(header_font_size, bold=True)

    if schedule_slots:
        box_x1, box_y1 = int(page_w * 0.07), int(page_h * 0.05)
        box_x2, box_y2 = page_w - int(page_w * 0.07), int(page_h * 0.15)
        draw.rounded_rectangle((box_x1, box_y1, box_x2, box_y2), radius=20, fill=(255, 255, 255, 220), outline=theme["accent"] + (200,), width=3)

        draw.text((box_x1 + 20, box_y1 + 15), "TODAY'S SCHEDULE / PERIODS", font=header_font, fill=theme["dark"])
        draw.text((box_x2 - int(page_w * 0.22), box_y1 + 15), "DATE: ____ / ____ / ____", font=header_font, fill=theme["dark"])

        cols = 4 if len(schedule_slots) >= 8 else 3
        col_w = (box_x2 - box_x1 - 30) // cols
        row_h = int(page_h * 0.018)
        slot_font = get_font(slot_font_size, bold=False)

        for idx, slot in enumerate(schedule_slots[:12]):
            c = idx % cols
            r = idx // cols
            sx = box_x1 + 15 + c * col_w
            sy = box_y1 + int(page_h * 0.025) + r * row_h
            
            draw.rectangle((sx, sy, sx + col_w - 10, sy + row_h - 5), fill=theme["background"] + (180,), outline=theme["accent"] + (120,), width=2)
            draw.text((sx + 8, sy + 4), str(slot), font=slot_font, fill=(60, 60, 60))

        line_start_y = box_y2 + int(page_h * 0.02)
    else:
        draw.rounded_rectangle((page_w - int(page_w * 0.28), int(page_h * 0.05), page_w - int(page_w * 0.07), int(page_h * 0.085)), radius=15, fill=(255, 255, 255, 200), outline=theme["accent"] + (200,), width=3)
        draw.text((page_w - int(page_w * 0.26), int(page_h * 0.06)), "DATE: ____ / ____ / ________", font=header_font, fill=theme["dark"])
        line_start_y = int(page_h * 0.11)

    # Lines Section
    line_end_y = page_h - int(page_h * 0.07)
    line_spacing = int(page_h * 0.025)
    margin_x = int(page_w * 0.07)

    for y in range(line_start_y, line_end_y, line_spacing):
        draw.line((margin_x, y, page_w - margin_x, y), fill=theme["line_color"] + (220,), width=3)

    # Red Margin Line
    draw.line((margin_x, line_start_y - 20, margin_x, page_h - int(page_h * 0.05)), fill=(240, 120, 120, 180), width=4)

    # Page Number
    footer_font = get_font(int(page_w * 0.015), bold=False)
    draw.text((page_w // 2 - 20, page_h - int(page_h * 0.04)), f"- {page_num} -", font=footer_font, fill=theme["dark"])

    return page.convert("RGB")

# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📚 Custom Printable Notebook Generator")
st.write("Create custom printable lined notebooks with watermarks, sparkling cover pages, and optional daily schedules!")

uploaded_files = st.file_uploader(
    "Choose drawing photos (JPG, PNG, WEBP):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

st.subheader("📏 Page Size & Notebook Format")
col_s1, col_s2 = st.columns(2)

with col_s1:
    chosen_size_label = st.selectbox("Select Page Size:", list(PAGE_SIZES.keys()))
    page_dimensions = PAGE_SIZES[chosen_size_label]
    PAGE_W, PAGE_H = page_dimensions["width"], page_dimensions["height"]

with col_s2:
    page_count_preset = st.selectbox(
        "Select Number of Notebook Pages:",
        ["Standard (20 Pages)", "Monthly Journal (30 Pages)", "Semester / Yearly (100 Pages)", "Custom Page Count"]
    )

if page_count_preset == "Standard (20 Pages)":
    TOTAL_PAGES = 20
elif page_count_preset == "Monthly Journal (30 Pages)":
    TOTAL_PAGES = 30
elif page_count_preset == "Semester / Yearly (100 Pages)":
    TOTAL_PAGES = 100
else:
    TOTAL_PAGES = st.number_input("Enter Custom Page Count (1 to 200):", min_value=1, max_value=200, value=15)

st.subheader("👤 Personalization & Profile")
col_a, col_b = st.columns(2)
with col_a:
    child_name = st.text_input("Notebook Owner Name:", "Alex")
with col_b:
    role_type = st.selectbox(
        "Who is this notebook for?",
        ["Standard User", "School Student", "Office Worker", "PhD Student / Researcher", "Custom"]
    )

role_title = ""
if role_type == "School Student":
    role_title = "School Student"
elif role_type == "Office Worker":
    role_title = "Office Worker"
elif role_type == "PhD Student / Researcher":
    role_title = "PhD Researcher"
elif role_type == "Custom":
    role_title = st.text_input("Custom Title/Role:", "Creative Notes")

# Optional Schedule
st.subheader("⏰ Daily Schedule / Periods Block (Optional)")
enable_schedule = st.checkbox("Include Schedule / Periods header on notebook pages?", value=False)

schedule_slots = []
if enable_schedule:
    schedule_mode = st.radio(
        "Choose Schedule Division Format:",
        ["Hourly Intervals (e.g. 10:00 - 11:00)", "Periods (e.g. Period 1, Period 2)", "Custom Time Slots"]
    )

    if schedule_mode == "Hourly Intervals (e.g. 10:00 - 11:00)":
        start_hour = st.number_input("Start Hour (24h format):", min_value=0, max_value=23, value=9)
        total_hours = st.slider("Total Working Hours:", min_value=2, max_value=12, value=8)
        for h in range(total_hours):
            h1 = (start_hour + h) % 24
            h2 = (start_hour + h + 1) % 24
            schedule_slots.append(f"{h1:02d}:00 - {h2:02d}:00")

    elif schedule_mode == "Periods (e.g. Period 1, Period 2)":
        num_periods = st.slider("Number of Periods:", min_value=2, max_value=10, value=8)
        period_length = st.selectbox("Period Duration:", ["40 mins", "45 mins", "50 mins", "60 mins"])
        for p in range(1, num_periods + 1):
            schedule_slots.append(f"P{p} ({period_length})")

    else:
        custom_input = st.text_area("Enter time slots separated by commas:", "10-11 AM, 11-12 PM, 12-1 PM, 2-3 PM, 3-4 PM")
        schedule_slots = [s.strip() for s in custom_input.split(",") if s.strip()]

if uploaded_files:
    if st.button(f"🚀 Generate {TOTAL_PAGES}-Page Notebook PDF", type="primary", use_container_width=True):
        processed_drawings = []

        with st.spinner("Extracting drawings and removing background paper..."):
            for file_obj in uploaded_files:
                raw_img = Image.open(file_obj)
                raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
                img = resize_image(raw_img, MAX_SIZE)
                clean_trans = create_transparent_drawing(img)
                processed_drawings.append(clean_trans)

        st.success(f"Processed {len(processed_drawings)} drawing(s) successfully!")

        with st.spinner(f"Building dynamic cover and {TOTAL_PAGES} interior pages ({chosen_size_label})..."):
            all_pdf_pages = []

            # Cover Page
            cover_seed = random.randint(1, 10000)
            dark_theme = random.choice(DARK_THEMES)
            cover_page = create_dynamic_cover_page(processed_drawings, child_name, role_title, dark_theme, cover_seed, PAGE_W, PAGE_H)
            all_pdf_pages.append(cover_page)

            # Interior Lined Pages
            for page_num in range(1, TOTAL_PAGES + 1):
                active_drawing = processed_drawings[(page_num - 1) % len(processed_drawings)]
                pastel_theme = PASTEL_THEMES[(page_num - 1) % len(PASTEL_THEMES)]
                seed = 1000 + page_num

                page_img = create_lined_notebook_page(
                    active_drawing, pastel_theme, page_num, seed, PAGE_W, PAGE_H,
                    schedule_slots=schedule_slots if enable_schedule else None
                )
                all_pdf_pages.append(page_img)

        # Previews
        st.subheader("🖼️ Page Previews")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.image(all_pdf_pages[0], caption="Dynamic Cover Page", use_container_width=True)
        with col_p2:
            st.image(all_pdf_pages[1], caption="Sample Interior Page (Page 1)", use_container_width=True)

        # Export PDF
        pdf_buf = io.BytesIO()
        all_pdf_pages[0].save(
            pdf_buf,
            "PDF",
            resolution=DPI,
            save_all=True,
            append_images=all_pdf_pages[1:]
        )

        st.subheader("📥 Download Your Printable PDF")
        st.download_button(
            label=f"Download Complete {TOTAL_PAGES + 1}-Page Notebook PDF",
            data=pdf_buf.getvalue(),
            file_name=f"{child_name}_Custom_Notebook_{TOTAL_PAGES}_Pages.pdf",
            mime="application/pdf",
            use_container_width=True
        )
