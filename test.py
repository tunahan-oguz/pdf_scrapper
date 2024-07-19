import fitz  # PyMuPDF
from PIL import Image, ImageDraw

# Function to convert drawing commands to an image
def draw_to_image(drawings, width, height):
    # Convert dimensions to integers
    width = int(width)
    height = int(height)
    
    # Create a blank white canvas
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Loop through each drawing object
    for drawing in drawings:
        for item in drawing['items']:
            if item[0] == 'l':  # line
                points = [(item[1][0], item[1][1]), (item[1][2], item[1][3])]
                color = (item[3], item[4], item[5])
                draw.line(points, fill=color, width=int(item[2]))
            elif item[0] == 're':  # rectangle
                rect = item[1]
                color = item[-1]
                draw.rectangle(rect, outline=color, width=int(item[5]))
            elif item[0] == 'be':  # Bezier curve
                points = [(item[1][i], item[1][i+1]) for i in range(0, len(item[1]), 2)]
                color = (item[3], item[4], item[5])
                draw.line(points, fill=color, width=int(item[2]))
            elif item[0] == 'po':  # polyline
                points = [(item[1][i], item[1][i+1]) for i in range(0, len(item[1]), 2)]
                color = (item[3], item[4], item[5])
                draw.line(points, fill=color, width=int(item[2]))
            # Add more drawing types as needed

    return image

# Open the PDF document
pdf_document = "18.pdf"
doc = fitz.open(pdf_document)

# Select a page
page_number = 32  # Example: first page
page = doc.load_page(page_number)

# Get drawing objects
drawings = page.get_drawings()

# Get page size
page_width, page_height = page.rect.width, page.rect.height

# Convert drawings to image
image = draw_to_image(drawings, page_width, page_height)

# Save the image
image.save("drawings_page_1.png")
