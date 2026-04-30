import cv2
import numpy as np
import os
import random
import glob
from PIL import Image, ImageDraw, ImageFont, ImageStat
import urllib.request

# Font paths from local folder
font_dir = r"c:\Users\us21m\Desktop\project\Fonts"
cursive_font_path = os.path.join(font_dir, "GreatVibes-Regular.ttf")
sanskrit_font_path = os.path.join(font_dir, "YatraOne-Regular.ttf")
cinzel_font_path = sanskrit_font_path # fallback to YatraOne since Cinzel isn't there
alexbrush_font_path = os.path.join(font_dir, "AlexBrush-Regular.ttf")

def get_files(directory):
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return files

base_dir = r"c:\Users\us21m\Desktop\project\dataset"
template_dir = os.path.join(base_dir, "template")
ganesha_dir = os.path.join(base_dir, "ganesha")
lover_dir = os.path.join(base_dir, "lover")
output_dir = r"c:\Users\us21m\Desktop\project\output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

templates = get_files(template_dir)
ganeshas = get_files(ganesha_dir)
lovers = get_files(lover_dir)

print(f"Templates: {len(templates)}, Ganeshas: {len(ganeshas)}, Lovers: {len(lovers)}")

def get_image_size(path):
    with Image.open(path) as img:
        return img.size

def paste_image_with_alpha(bg, fg_path, x, y, scale=1.0):
    try:
        fg = Image.open(fg_path).convert("RGBA")
        if scale != 1.0:
            new_size = (int(fg.width * scale), int(fg.height * scale))
            fg = fg.resize(new_size, Image.Resampling.LANCZOS)
        
        # Fix coordinates if they go out of bounds
        if x < 0: x = 0
        if y < 0: y = 0
        if x + fg.width > bg.width:
            x = bg.width - fg.width
        if y + fg.height > bg.height:
            y = bg.height - fg.height
            
        bg.paste(fg, (x, y), fg)
        return fg.width, fg.height
    except Exception as e:
        print(f"Error pasting {fg_path}: {e}")
        return 0, 0

def find_best_placement(dist, w, h, region, element_w, element_h):
    mask = np.zeros_like(dist, dtype=np.uint8)
    if region == 'top':
        mask[int(h*0.02):int(h*0.3), int(w*0.2):int(w*0.8)] = 1
    elif region == 'bottom_right':
        mask[int(h*0.6):int(h*0.95), int(w*0.5):int(w*0.95)] = 1
    elif region == 'bottom_left':
        mask[int(h*0.6):int(h*0.95), int(w*0.05):int(w*0.5)] = 1
    elif region == 'bottom_center':
        mask[int(h*0.7):int(h*0.95), int(w*0.2):int(w*0.8)] = 1
    elif region == 'center':
        mask[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)] = 1
        
    masked_dist = dist * mask
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_dist)
    
    if max_val < 5: # Fallback if no free space found
        if region == 'top':
            return (w - element_w) // 2, int(h * 0.05)
        elif region == 'bottom_right':
            return w - element_w - int(w * 0.05), h - element_h - int(h * 0.05)
        elif region == 'bottom_left':
            return int(w * 0.05), h - element_h - int(h * 0.05)
        elif region == 'bottom_center':
            return (w - element_w) // 2, h - element_h - int(h * 0.05)
            
    center_x, center_y = max_loc
    return max(0, min(w - element_w, center_x - element_w // 2)), max(0, min(h - element_h, center_y - element_h // 2))

def get_contrasting_text_color(bg_img, y_start, y_end):
    try:
        w, h = bg_img.size
        y1 = max(0, int(y_start))
        y2 = min(h, int(y_end))
        if y2 <= y1: return (255, 215, 0)
        crop = bg_img.crop((0, y1, w, y2))
        stat = ImageStat.Stat(crop)
        r, g, b, _ = stat.mean
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        if luminance > 128:
            return (20, 20, 20) # Dark text for light bg
        else:
            return (255, 215, 0) # Gold text for dark bg
    except Exception as e:
        print(f"Color contrast error: {e}")
        return (255, 215, 0)

def get_fitting_font(text, font_path, max_width, initial_size, draw):
    size = initial_size
    font = ImageFont.truetype(font_path, size)
    while size > 10:
        try:
            bbox = draw.textbbox((0,0), text, font=font)
            tw = bbox[2] - bbox[0]
        except AttributeError:
            tw, _ = draw.textsize(text, font=font)
        if tw <= max_width:
            break
        size -= 2
        font = ImageFont.truetype(font_path, size)
    return font

def run_pipeline(bride_name="Jane", groom_name="John", date="12/12/2026", venue="Grand Palace"):
    if not templates or not ganeshas or not lovers:
        print("Missing dataset files.")
        return

    # Removed fixed seed so it's truly random every time

    for variant in range(1, 6):
        print(f"\n--- Generating Variant {variant} ---")
        
        template_path = random.choice(templates)
        ganesha_path = random.choice(ganeshas)
        lover_path = random.choice(lovers)
            
        # Stage 1: Ingestion
        print("Stage 1: Image Ingestion & Preprocessing")
        img_cv = cv2.imread(template_path)
        if img_cv is None:
            print(f"Failed to load {template_path}")
            continue
        h, w = img_cv.shape[:2]
        
        # Stage 2: Edge Detection
        print("Stage 2: Edge Detection & Structure Analysis")
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Stage 3: Free Space Mapping
        print("Stage 3: Free Space Detection & Zone Mapping")
        kernel = np.ones((5,5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        dist = cv2.distanceTransform(255 - dilated_edges, cv2.DIST_L2, 3)
        
        # Stage 4: Layout Plan Generation
        print("Stage 4: Layout Plan Generation")
        bg = Image.open(template_path).convert("RGBA")
        draw = ImageDraw.Draw(bg)
        
        ganesha_w, ganesha_h = get_image_size(ganesha_path)
        lover_w, lover_h = get_image_size(lover_path)
        
        # Make Ganesha much larger to grab attention
        ganesha_scale = (w * 0.25) / ganesha_w
        scaled_gw = int(ganesha_w * ganesha_scale)
        scaled_gh = int(ganesha_h * ganesha_scale)
        
        # Place Ganesha at top center but avoiding borders using distance map
        gx, gy = find_best_placement(dist, w, h, 'top', scaled_gw, scaled_gh)
        paste_image_with_alpha(bg, ganesha_path, gx, gy, scale=ganesha_scale)
        
        # We might not place the lover sticker if it clutters the text, but let's place it at bottom if space exists
        # Make the lover sticker much larger to grab attention
        lover_scale = (w * 0.40) / lover_w
        scaled_lw = int(lover_w * lover_scale)
        scaled_lh = int(lover_h * lover_scale)
        lx, ly = find_best_placement(dist, w, h, 'bottom_center', scaled_lw, scaled_lh)
        
        place_lover = random.choice([True, False])
        if place_lover:
            paste_image_with_alpha(bg, lover_path, lx, ly, scale=lover_scale)
        else:
            ly = h - int(h * 0.05) # If no lover sticker, text can go lower

        # Find the safe vertical span for text
        safe_top_y = gy + scaled_gh + int(h * 0.02)
        safe_bottom_y = ly - int(h * 0.02)
        available_height = safe_bottom_y - safe_top_y
            
        # Get dynamic contrasting color
        text_color = get_contrasting_text_color(bg, safe_top_y, safe_bottom_y)
        
        def get_text_h(text, font_path, max_w, size):
            font = get_fitting_font(text, font_path, max_w, size, draw)
            try:
                bbox = draw.textbbox((0,0), text, font=font)
                return bbox[3] - bbox[1]
            except AttributeError:
                _, th = draw.textsize(text, font=font)
                return th

        max_text_width = int(w * 0.8) # Keep some padding

        font_path_small = cinzel_font_path if random.choice([True, False]) else sanskrit_font_path
        invitation_text = random.choice(["Wedding Invitation", "Save The Date"])
        font_path_cursive = random.choice([alexbrush_font_path, cursive_font_path])
        font_path_names = random.choice([alexbrush_font_path, cursive_font_path, sanskrit_font_path])
        font_path_medium = cinzel_font_path if random.choice([True, False]) else sanskrit_font_path
        
        is_stacked = random.choice([True, False])

        # Simulate total height
        sim_h = 0
        sim_h += get_text_h("|| Shri Ganeshaya Namah ||", font_path_small, max_text_width, int(w*0.03)) + int(h * 0.02)
        sim_h += int(h * 0.02)
        sim_h += get_text_h(invitation_text, font_path_small, max_text_width, int(w*0.04)) + int(h * 0.02)
        sim_h += int(h * 0.03)
        if is_stacked:
            sim_h += get_text_h(bride_name.capitalize(), font_path_names, max_text_width, int(w*0.18)) + int(h * 0.02)
            sim_h -= int(h * 0.02)
            sim_h += get_text_h("and", font_path_cursive, max_text_width, int(w*0.05)) + int(h * 0.02)
            sim_h -= int(h * 0.02)
            sim_h += get_text_h(groom_name.capitalize(), font_path_names, max_text_width, int(w*0.18)) + int(h * 0.02)
        else:
            sim_h += get_text_h(f"{bride_name} & {groom_name}", font_path_names, max_text_width, int(w*0.15)) + int(h * 0.02)
        sim_h += int(h * 0.04)
        sim_h += get_text_h(date.upper(), font_path_medium, max_text_width, int(w*0.035)) + int(h * 0.02)
        sim_h += int(h * 0.01)
        sim_h += get_text_h(venue.upper(), font_path_small, max_text_width, int(w*0.025)) + int(h * 0.02)

        font_mult = 1.0
        if sim_h > available_height:
            font_mult = (available_height * 0.9) / sim_h # 90% to be safe
            
        # Center vertically if there is space
        if available_height > (sim_h * font_mult):
            current_y = safe_top_y + (available_height - (sim_h * font_mult)) // 2
        else:
            current_y = safe_top_y

        def draw_centered_text(text, font_path, max_w, base_size, y_pos, color=text_color):
            size = max(10, int(base_size * font_mult))
            font = get_fitting_font(text, font_path, max_w, size, draw)
            try:
                bbox = draw.textbbox((0,0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
            tx = (w - tw) // 2
            draw.text((tx, y_pos), text, fill=color, font=font)
            return y_pos + th + int(h * 0.02 * font_mult)

        # Draw structured text
        # 1. || Shri Ganeshaya Namah ||
        current_y = draw_centered_text("|| Shri Ganeshaya Namah ||", font_path_small, max_text_width, int(w*0.03), current_y)
        
        # 2. Add some space
        current_y += int(h * 0.02 * font_mult)
        
        # 3. Wedding Invitation / Save The Date
        current_y = draw_centered_text(invitation_text, font_path_small, max_text_width, int(w*0.04), current_y)
        
        # 4. Add some space
        current_y += int(h * 0.03 * font_mult)
        
        if is_stacked:
            current_y = draw_centered_text(bride_name.capitalize(), font_path_names, max_text_width, int(w*0.18), current_y)
            current_y -= int(h * 0.02 * font_mult) # Tighter spacing
            current_y = draw_centered_text("and", font_path_cursive, max_text_width, int(w*0.05), current_y)
            current_y -= int(h * 0.02 * font_mult)
            current_y = draw_centered_text(groom_name.capitalize(), font_path_names, max_text_width, int(w*0.18), current_y)
        else:
            names_text = f"{bride_name} & {groom_name}"
            current_y = draw_centered_text(names_text, font_path_names, max_text_width, int(w*0.15), current_y)
        
        # 6. Date & Time
        current_y += int(h * 0.04 * font_mult)
        current_y = draw_centered_text(date.upper(), font_path_medium, max_text_width, int(w*0.035), current_y)
        
        # 7. Venue
        current_y += int(h * 0.01 * font_mult)
        current_y = draw_centered_text(venue.upper(), font_path_small, max_text_width, int(w*0.025), current_y)
        
        out_path = os.path.join(output_dir, f"variant_{variant}.jpg")
        bg.convert("RGB").save(out_path, quality=95)
        print(f"Saved {out_path}")

if __name__ == '__main__':
    run_pipeline()
