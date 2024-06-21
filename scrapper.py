import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import os
import tqdm
from utils import *
import glob
import post_process_texts
import train.vocab

#CHANGE THE PATH
pdfs = glob.glob("/home/oguz/Desktop/BIL471/project/anatomybooks/*.pdf")

def extract_images_and_descriptions(pdf_path):
    global IMAGE_ROOT
    i_root = os.path.join(IMAGE_ROOT, os.path.basename(pdf_path))
    os.makedirs(i_root, exist_ok=True)
    all_text_boxes = get_text_boxes(pdf_path)
    all_text_boxes = pre_process_text_boxes(all_text_boxes)

    pdf_document = fitz.open(pdf_path)
    image_data = []

    for page_number in tqdm.tqdm(range(len(pdf_document))):
        page = pdf_document.load_page(page_number)
        images = page.get_images(full=True)
        for image_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            if base_image["width"] > 50 and base_image["height"] > 50:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Save the image
                    image_filename = f"page_{page_number+1}_image_{image_index+1}.{image_ext}"
                    image_file_path = os.path.join(i_root, image_filename)
                    
                    # Extract text blocks and find those immediately below the image
                    text_blocks = page.get_text("blocks")
                    
                    bbox = page.get_image_rects(xref)[0]
                    image_center_x = (bbox.x0 + bbox.x1) / 2
                    image_center_y = (bbox.y0 + bbox.y1) / 2

                    min_distance = float('inf')
                    caption_text = ""
                    image_kwd = None
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

                    if image_kwd is None: continue
                    formatter = FIGURE_TO_REF_FORMATS[os.path.basename(pdf_path)]
                    image_kwd = formatter(image_kwd) if formatter is not None else None

                    instance = (image_file_path, caption_text, get_refs(all_text_boxes, image_kwd))
                    image.save(image_file_path)
                    image_data.append(instance)
                except Exception as e:
                    print(f"Error processing image on page {page_number+1}: {e}")
    return image_data

if __name__ == "__main__":
    for pdf_path in pdfs:
        print(f"Processing book named {os.path.basename(pdf_path)}")
        image_data = extract_images_and_descriptions(pdf_path)
        df = pd.DataFrame(image_data, columns=["Image", "Description", "Reference"])
        os.makedirs(DESC_ROOT, exist_ok=True)
        df.to_csv(f"{DESC_ROOT}/image_descriptions{os.path.basename(pdf_path)}.csv", index=False)
        # df.to_json(f"{DESC_ROOT}/image_descriptions{os.path.basename(pdf_path)}.json", orient="records")
    post_process_texts.main()
    train.vocab.main('dataset/descriptions', 'simple')
