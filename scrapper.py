import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import os
from rich.progress import Progress
import tqdm

IMAGE_ROOT = "dataset/images"
DESC_ROOT = "dataset/descriptions"
KWD = "Figure"
pdf_path = "/home/oguz/Desktop/BIL471/project/anatomybooks/7.pdf"
jump = 13
stop = 140

def extract_images_and_descriptions(pdf_path, start, stop):
    global IMAGE_ROOT
    IMAGE_ROOT = os.path.join(IMAGE_ROOT, os.path.basename(pdf_path))
    os.makedirs(IMAGE_ROOT, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    image_data = []
    
    for page_number in tqdm.tqdm(range(start, stop)):
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
                    image_file_path = os.path.join(IMAGE_ROOT, image_filename)
                    image.save(image_file_path)
                    
                    # Extract text blocks and find those immediately below the image
                    text_blocks = page.get_text("blocks")
                    caption_text = ""
                    
                    bbox = page.get_image_rects(xref)[0]
                    image_center_x = (bbox.x0 + bbox.x1) / 2
                    image_center_y = (bbox.y0 + bbox.y1) / 2

                    min_distance = float('inf')
                    
                    for block in text_blocks:
                        if not block[4].startswith(KWD):
                            continue
                        block_bbox = fitz.Rect(block[:4])
                        block_center_x = (block_bbox.x0 + block_bbox.x1) / 2
                        block_center_y = (block_bbox.y0 + block_bbox.y1) / 2
                        
                        # Calculate distance from image center to block center
                        distance = ((block_center_x - image_center_x) ** 2 + (block_center_y - image_center_y) ** 2) ** 0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            caption_text = block[4]
                
                    # Store image path and corresponding caption text
                    image_data.append((image_file_path, caption_text.strip()))
                except Exception as e:
                    print(f"Error processing image on page {page_number+1}: {e}")
    return image_data


image_data = extract_images_and_descriptions(pdf_path, jump, stop)


df = pd.DataFrame(image_data, columns=["Image", "Description"])
os.makedirs(DESC_ROOT, exist_ok=True)
df.to_csv(f"{DESC_ROOT}/image_descriptions{os.path.basename(pdf_path)}.csv", index=False)
df.to_json(f"{DESC_ROOT}/image_descriptions{os.path.basename(pdf_path)}.json", orient="records")
