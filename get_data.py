import urllib.request
import zipfile

# --------------- REACTION FILES
zip_file    = 'DATA/cifs.zip'
urllib.request.urlretrieve("https://figshare.com/ndownloader/files/34195719", zip_file)

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('DATA/')
# --------------------------------------------------------------------------------------


print('Done extracting the files')
