import io
import urllib

from PIL import Image
from urllib.request import urlopen

def valid_coord(text):
    args = text.split()
    
    if len(args) != 3:
        return None

    try:
        lat = float(args[0])
        lon = float(args[1])
        zoom = int(args[2])
    except Exception as e:
        return None

    return lat, lon, zoom

def get_map_picture(lat, lon, zoom=14):
    url = f'https://static-maps.yandex.ru/1.x/?ll={lon},{lat}&size=450,450&z={zoom}&l=sat'
    try:
        url_response = urlopen(url)
        
        if url_response.status == 200:
            imageStream = io.BytesIO(url_response.read())
            imageFile = Image.open(imageStream)
            height, width = imageFile.size
            imageFile = imageFile.crop((0, 0, width, height - 50))

            #imageStream.close()
            return imageFile
        else:
            return None
    except urllib.error.HTTPError as e:
        return None

if __name__ == '__main__':
    get_map_picture(37.620070, 55.753630).show()