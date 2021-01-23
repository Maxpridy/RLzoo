import base64

pt_string = ""

with open('model.pt', 'rb') as f:
    pt_string = f.read()

print(pt_string)

encoded = base64.b64encode(pt_string)
print(encoded)
decoded = base64.b64decode(encoded)
print(decoded)

print(pt_string == decoded)