from PIL import Image

bild = "hallo1"
myImage = Image.open(f"images\input_Images\{bild}.png")
rotated_image = myImage.rotate(90, expand=True)
rotated_image.show()

rotated_image.save(f"images\processed_Images\{bild}p1.png")

