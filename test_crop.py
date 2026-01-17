from PIL import Image
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from segment_agent.skills.tools.img_tool.img_tool import crop_img

def test_crop_oob():
    # Create a 100x100 dummy image
    img = Image.new('RGB', (100, 100), color='red')
    
    print(f"Original Size: {img.size}")
    
    # Try to crop 0,0 to 150,150
    try:
        cropped = crop_img(img, (0, 0), (150, 150))
        print(f"OOB Crop (0,0,150,150) Result Size: {cropped.size}")
        # Check if it physically expanded or clipped
        print(f"Is 0,0 pixel present? {cropped.getpixel((0,0))}")
        try:
             print(f"Is 149,149 pixel present? {cropped.getpixel((149,149))}")
        except Exception as e:
             print(f"Accessing 149,149 failed: {e}")
             
    except Exception as e:
        print(f"OOB Crop failed with error: {e}")

if __name__ == "__main__":
    test_crop_oob()
