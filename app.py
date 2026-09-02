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
    page_title="Cute Notebook Generator",
    page_icon="📚",
    layout="centered"
)

# ============================================================
# PRINT SETTINGS (A4 @ 300 DPI)
# ============================================================

A4_WIDTH = 2480
A4_HEIGHT = 3508
DPI = 300
MAX_SIZE = 1800

AUTO_STRAIGHTEN_DRAWING = True
MAX_STRAIGHTEN_ANGLE = 18

ADD_DECORATIONS = True
ADD_STARS = True
ADD_HEARTS = True
ADD_FLOWERS = True
ADD_SPARKLES = True
ADD_GLITTER = True

STAR_COUNT = 8
HEART_COUNT = 6
FLOWER_COUNT = 6
SPARKLE_COUNT = 10
GLITTER_COUNT = 40

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
        "background": (35, 38, 75),
        "accent": (255, 190, 80),
        "text": (255, 255, 255),
        "shape1": (90, 70, 160),
        "shape2": (45, 110, 180)
    },
    {
        "name": "Ocean Night",
        "background": (25, 70, 95),
        "accent": (255, 210, 90),
        "text": (255, 255, 255),
        "shape1": (40, 150, 180),
        "shape2": (80, 190, 200)
    },
    {
        "name": "Royal Purple",
        "background": (65, 35, 105),
        "accent": (255, 195, 100),
        "text": (255, 255, 255),
        "shape1": (145, 100, 210),
        "shape2": (200, 160, 240)
    },
    {
        "name": "Berry Pop",
        "background": (110, 35, 80),
        "accent": (255, 205, 100),
        "text": (255, 255, 255),
        "shape1": (210, 80, 135),
        "shape2": (245, 150, 180)
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
# DECORATIONS & DRAWING HELPERS
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

def draw_heart(draw, x, y, size, color):
    points = [
        (x, y + size),
        (x - size, y),
        (x - size * 0.8, y - size * 0.7),
        (x - size * 0.35, y - size),
        (x, y - size * 0.5),
        (x + size * 0.35, y - size),
        (x + size * 0.8, y - size * 0.7),
        (x + size, y)
    ]
    draw.polygon(points, fill=color)

def draw_sparkle(draw, x, y, size, color):
    width = max(2, int(size * 0.16))
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)

def add_decorations(page, theme, rng):
    draw = ImageDraw.Draw(page, "RGBA")
    w, h = page.size
    accent = theme["accent"]

    for _ in range(STAR_COUNT):
        x = rng.choice([rng.randint(80, 300), rng.randint(w - 300, w - 80)])
        y = rng.randint(100, h - 100)
        draw_star(draw, x, y, rng.randint(25, 55), accent + (180,))

    for _ in range(HEART_COUNT):
        x = rng.choice([rng.randint(80, 300), rng.randint(w - 300, w - 80)])
        y = rng.randint(100, h - 100)
        draw_heart(draw, x, y, rng.randint(20, 45), accent + (160,))

    for _ in range(SPARKLE_COUNT):
        x = rng.randint(100, w - 100)
        y = rng.randint(100, h - 100)
        draw_sparkle(draw, x, y, rng.randint(15, 35), (255, 255, 255, 200))

    return page

# ============================================================
# NOTEBOOK PAGE BUILDERS
# ============================================================

def create_lined_notebook_page(transparent_drawing, theme, page_num, seed):
    rng = random.Random(seed + page_num)
    page = Image.new("RGBA", (A4_WIDTH, A4_HEIGHT), theme["background"] + (255,))

    # Add background shapes
    draw = ImageDraw.Draw(page, "RGBA")
    draw.ellipse((-300, -200, 800, 800), fill=theme["shape1"] + (120,))
    draw.ellipse((A4_WIDTH - 700, A4_HEIGHT - 700, A4_WIDTH + 300, A4_HEIGHT + 300), fill=theme["shape2"] + (120,))

    # Add background decorations
    page = add_decorations(page, theme, rng)

    # 1. Add Faded Watermark Drawing into Background
    if transparent_drawing:
        faded_drawing = transparent_drawing.copy()
        
        # Scale drawing to fit middle watermark area
        max_wm_w, max_wm_h = int(A4_WIDTH * 0.65), int(A4_HEIGHT * 0.45)
        scale = min(max_wm_w / faded_drawing.width, max_wm_h / faded_drawing.height)
        new_wm_size = (int(faded_drawing.width * scale), int(faded_drawing.height * scale))
        faded_drawing = faded_drawing.resize(new_wm_size, Image.Resampling.LANCZOS)

        # Lower opacity to 20% (watermark)
        alpha = faded_drawing.getchannel("A")
        alpha = alpha.point(lambda p: int(p * 0.22))
        faded_drawing.putalpha(alpha)

        wm_x = (A4_WIDTH - faded_drawing.width) // 2
        wm_y = (A4_HEIGHT - faded_drawing.height) // 2
        page.alpha_composite(faded_drawing, (wm_x, wm_y))

    # 2. Draw Writing Lines
    draw = ImageDraw.Draw(page, "RGBA")
    line_start_y = 450
    line_end_y = A4_HEIGHT - 300
    line_spacing = 90
    margin_x = 220

    for y in range(line_start_y, line_end_y, line_spacing):
        draw.line((margin_x, y, A4_WIDTH - margin_x, y), fill=theme["line_color"] + (220,), width=3)

    # Red vertical margin line
    draw.line((margin_x, 350, margin_x, A4_HEIGHT - 200), fill=(240, 120, 120, 180), width=4)

    # Header Date Box
    draw.rounded_rectangle((A4_WIDTH - 650, 200, A4_WIDTH - 220, 320), radius=20, outline=theme["accent"] + (200,), width=3)
    header_font = get_font(38, bold=True)
    draw.text((A4_WIDTH - 620, 235), "DATE: ____ / ____ / ________", font=header_font, fill=theme["dark"])

    # Page Number Footer
    footer_font = get_font(36, bold=False)
    draw.text(((A4_WIDTH) // 2 - 40, A4_HEIGHT - 180), f"- {page_num} -", font=footer_font, fill=theme["dark"])

    return page.convert("RGB")

def create_dark_cover_page(transparent_drawings, child_name, theme):
    rng = random.Random(101)
    cover = Image.new("RGBA", (A4_WIDTH, A4_HEIGHT), theme["background"] + (255,))
    draw = ImageDraw.Draw(cover, "RGBA")

    # Dark background accent shapes
    draw.ellipse((-400, -300, 1200, 1100), fill=theme["shape1"] + (140,))
    draw.ellipse((A4_WIDTH - 1100, A4_HEIGHT - 1200, A4_WIDTH + 400, A4_HEIGHT + 400), fill=theme["shape2"] + (140,))

    # Title Block
    title_font = get_font(110, bold=True)
    subtitle_font = get_font(60, bold=False)
    
    title_text = "MY CREATIVE NOTEBOOK"
    bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_w = bbox[2] - bbox[0]
    draw.text(((A4_WIDTH - title_w) // 2, 380), title_text, font=title_font, fill=theme["accent"])

    if child_name:
        name_text = f"Created by: {child_name}"
        bbox_n = draw.textbbox((0, 0), name_text, font=subtitle_font)
        name_w = bbox_n[2] - bbox_n[0]
        draw.text(((A4_WIDTH - name_w) // 2, 530), name_text, font=subtitle_font, fill=theme["text"])

    # Position All Images Together in Cover Center
    if transparent_drawings:
        count = len(transparent_drawings)
        max_item_w = int(A4_WIDTH * 0.40)
        max_item_h = int(A4_HEIGHT * 0.35)

        positions = [
            (A4_WIDTH // 2 - max_item_w // 2, A4_HEIGHT // 2 - max_item_h // 2),
            (A4_WIDTH // 4 - 100, A4_HEIGHT // 2 - 250),
            (3 * A4_WIDTH // 4 - max_item_w + 100, A4_HEIGHT // 2 - 250),
            (A4_WIDTH // 4 - 100, A4_HEIGHT // 2 + 250),
            (3 * A4_WIDTH // 4 - max_item_w + 100, A4_HEIGHT // 2 + 250)
        ]

        for i, img in enumerate(transparent_drawings[:5]):
            resized_img = fit_drawing_cover(img, max_item_w, max_item_h)
            pos = positions[i if i < len(positions) else 0]
            cover.alpha_composite(resized_img, pos)

    # Decorative frame line
    draw.rounded_rectangle((80, 80, A4_WIDTH - 80, A4_HEIGHT - 80), radius=50, outline=theme["accent"] + (200,), width=10)

    return cover.convert("RGB")

def fit_drawing_cover(drawing, max_width, max_height):
    w, h = drawing.size
    scale = min(max_width / w, max_height / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return drawing.resize(new_size, Image.Resampling.LANCZOS)

# ============================================================
# STREAMLIT UI
# ============================================================

st.title("📚 20-Page Custom Lined Notebook Generator")
st.write("Upload drawing photos to create a 20-page printable lined notebook with faded background watermarks and a dark cover page!")

uploaded_files = st.file_uploader(
    "Choose drawing photos (JPG, PNG, WEBP):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

child_name = st.text_input("Notebook Owner Name (e.g. Alex's Notebook):", "")

if uploaded_files:
    if st.button("🚀 Generate 20-Page Notebook PDF", type="primary", use_container_width=True):
        processed_drawings = []

        with st.spinner("Extracting drawings and cleaning backgrounds..."):
            for file_obj in uploaded_files:
                raw_img = Image.open(file_obj)
                raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
                img = resize_image(raw_img, MAX_SIZE)
                clean_trans = create_transparent_drawing(img)
                processed_drawings.append(clean_trans)

        st.success(f"Processed {len(processed_drawings)} drawing(s) successfully!")

        with st.spinner("Building 20 interior notebook pages and dark cover..."):
            all_pdf_pages = []

            # 1. Build Cover Page (Dark Theme)
            dark_theme = DARK_THEMES[0]
            cover_page = create_dark_cover_page(processed_drawings, child_name, dark_theme)
            all_pdf_pages.append(cover_page)

            # 2. Build 20 Interior Lined Pages
            for page_num in range(1, 21):
                # Assign a drawing sequentially or randomly
                active_drawing = processed_drawings[(page_num - 1) % len(processed_drawings)]
                pastel_theme = PASTEL_THEMES[(page_num - 1) % len(PASTEL_THEMES)]
                seed = 500 + page_num

                page_img = create_lined_notebook_page(active_drawing, pastel_theme, page_num, seed)
                all_pdf_pages.append(page_img)

        st.image(all_pdf_pages[0], caption="Cover Page Preview", use_container_width=True)
        st.image(all_pdf_pages[1], caption="Sample Interior Lined Page (Page 1)", use_container_width=True)

        # 3. Export PDF
        pdf_buf = io.BytesIO()
        all_pdf_pages[0].save(
            pdf_buf,
            "PDF",
            resolution=DPI,
            save_all=True,
            append_images=all_pdf_pages[1:]
        )

        st.subheader("📥 Download Your Printable Notebook")
        st.download_button(
            label="Download Complete 21-Page Notebook PDF (Cover + 20 Pages)",
            data=pdf_buf.getvalue(),
            file_name="Custom_Lined_Notebook_20_Pages.pdf",
            mime="application/pdf",
            use_container_width=True
        )
