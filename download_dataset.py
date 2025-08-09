import deeplake

print("Downloading FER2013 training dataset...")
ds_train = deeplake.open("hub://activeloop/fer2013-train")
print("FER2013 training dataset downloaded.")

print("Downloading FER2013 public test dataset...")
ds_public_test = deeplake.open("hub://activeloop/fer2013-public-test")
print("FER2013 public test dataset downloaded.")

print("Downloading FER2013 private test dataset...")
ds_private_test = deeplake.open("hub://activeloop/fer2013-private-test")
print("FER2013 private test dataset downloaded.")


