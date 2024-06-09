import fitz  # PyMuPDF

def draw_bounding_rects_on_pdf(input_pdf_path, output_pdf_path):
    # Open the PDF
    pdf_document = fitz.open(input_pdf_path)
    
    # Iterate through each page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        
        # Get text blocks
        text_blocks = page.get_text("blocks")
        
        # Draw rectangles around text blocks
        for block in text_blocks:
            block_bbox = fitz.Rect(block[:4])
            page.draw_rect(block_bbox, color=(0, 0, 1), width=0.5)  # Blue rectangles for text blocks
        
        # Get image blocks
        images = page.get_images(full=True)
        
        # Draw rectangles around images
        for img in images:
            bbox = page.get_image_rects(img[0])[0]
            page.draw_rect(bbox, color=(1, 0, 0), width=0.5)  # Red rectangles for images
    
    # Save the modified PDF
    pdf_document.save(output_pdf_path)

# Define the input and output PDF file paths
input_pdf_path = "/home/oguz/Desktop/BIL471/anatomybooks/1.pdf"
output_pdf_path = "output_with_bounding_rects.pdf"

# Draw bounding rectangles on the PDF
draw_bounding_rects_on_pdf(input_pdf_path, output_pdf_path)
