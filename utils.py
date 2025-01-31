import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import re
from nltk.tokenize import word_tokenize

IMAGE_ROOT = "dataset/images"
DESC_ROOT = "dataset/descriptions"
# PATTERN = re.compile(r'\(?\b(?:Figure|FIGURE|Fig\.|FIG\.)\s*([0-9]+(?:\.[0-9]+)*(?:\.[0-9]+)?(?:-[0-9]+)?)\.?\)?', re.IGNORECASE)
PATTERN = re.compile(r'\b(?:FIG\.|Figure|Fig\.)\s+(\d{1,3}(?:[-.\u2013]\d{1,3}){0,2})\b', re.IGNORECASE)

def get_text_boxes(pdf_path):
    document = fitz.open(pdf_path)
    boxes = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("blocks")  # Extract text blocks
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # Sort blocks by vertical and then horizontal position
        for block in blocks:
            boxes.append(block[4].replace("\u2002", "")) # Append the text of the block and a newline for separation
    return boxes

def extract_text_from_pdf(pdf_path):
    # Extract text from the PDF
    text = extract_text(pdf_path)
    return text

def pre_process_text_boxes(all_text_boxes: list[str]) -> list[str]:
    processed_text_boxes = [text_box for text_box in all_text_boxes if PATTERN.search(text_box)]
    return processed_text_boxes

def get_refs(all_text_boxes: list[str], image_kwd: str) -> list[str]:
    if image_kwd is None: return []
    pattern = re.compile(r'\b{}\b'.format(re.escape(image_kwd)), re.IGNORECASE)
    refs = [text_box for text_box in all_text_boxes if pattern.search(text_box)]
    refs = [r for r in refs if not r.lower().startswith(image_kwd.lower())]
    return refs


def same(x):
    return x

def ninedotpdf(x):
    """Take the other text box in the page as reference.
    """
    return x

def figdot(image_kwd : str):
    return image_kwd.lower().replace("figure", "fig.")

def figdottofigure(image_kwd : str):
    return image_kwd.lower().replace("fig.", "figure")


FIGURE_TO_REF_FORMATS = {
    "1.pdf" : same,
    "2.pdf" : same,
    "3.pdf" : same,
    "4.pdf" : same,
    "5.pdf" : same,
    "6.pdf" : None,
    "7.pdf" : figdot,
    "8.pdf" : same,
    "9.pdf" : ninedotpdf,
    "10.pdf" : same,
    "11.pdf" : figdot,
    "12.pdf" : same,
    "13.pdf" : figdot,
    "14.pdf" : same,
    "15.pdf" : figdottofigure,
    "16.pdf" : same,
    "17.pdf" : same,
    "18.pdf" : same,
    "19.pdf" : None,
    "20.pdf" : same,
    "21.pdf" : figdot,
    "22.pdf" : figdot, #INVERSE CALISTIRMAK GEREKIYOR, BROKEN DATA STREAM WHEN WRITING IMAGE FILE ERROR ATIYOR COGU IMAGEDA}
}

def match(text_blocks, image_bbox):
    min_distance = float('inf')
    caption_text = ""
    image_kwd = None
    image_center_x = (image_bbox.x0 + image_bbox.x1) / 2
    image_center_y = (image_bbox.y0 + image_bbox.y1) / 2
    for block in text_blocks:
        match = PATTERN.search(block[4])
        if match is None:
            continue
        if not block[4].lower().startswith(match.group(0).lower()): continue
        block_bbox = fitz.Rect(block[:4])
        block_center_x = (block_bbox.x0 + block_bbox.x1) / 2
        block_center_y = (block_bbox.y0 + block_bbox.y1) / 2
        
        # Calculate distance from image center to block center
        distance = ((block_center_x - image_center_x) ** 2 + (block_center_y - image_center_y) ** 2) ** 0.5
        if block_center_y < image_center_y: continue # the caption must be below image
        if distance < min_distance:
            min_distance = distance
            caption_text = block[4]
            image_kwd = match.group(0)

    if image_kwd is None:
        return match2(text_blocks, image_bbox)
    
    return image_kwd, caption_text

def match2(text_blocks, image_bbox):
    min_distance = float('inf')
    caption_text = ""
    image_kwd = None
    image_center_x = (image_bbox.x0 + image_bbox.x1) / 2
    image_center_y = (image_bbox.y0 + image_bbox.y1) / 2
    for block in text_blocks:
        match = PATTERN.search(block[4])
        if match is None:
            continue
        if not block[4].lower().startswith(match.group(0).lower()): continue
        block_bbox = fitz.Rect(block[:4])
        block_center_x = (block_bbox.x0 + block_bbox.x1) / 2
        block_center_y = (block_bbox.y0 + block_bbox.y1) / 2
        
        # Calculate distance from image center to block center
        distance = ((block_center_x - image_center_x) ** 2 + (block_center_y - image_center_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            caption_text = block[4]
            image_kwd = match.group(0)
    
    return image_kwd, caption_text

def ref_crop(ref : str, tokenizer):
    words = tokenizer(ref)


def tag_ref(ref : str, m):
    while True:
        current_m = PATTERN.search(ref)
        if current_m is None: break
        ref = ref.replace(current_m.group(0), "" if current_m.group(0).lower() != m.lower() else "_T_")
    
    words = word_tokenize(ref)
    ref_point = -1
    for i, w in enumerate(words):
        if "_T_" in w:
            ref_point = i
            break
    
    if ref_point == -1:
        return ref
    words = [w for w in words if "_T_" not in w]
    sent = " ".join(words[max(0, ref_point - 7) : ref_point + 7])
    return sent.strip()