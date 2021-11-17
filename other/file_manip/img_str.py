# Convert image into a Base64 string
import base64

with open("seeds.jpg", "rb") as f:
	string = base64.b64encode(f.read())

print(string)
