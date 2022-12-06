import cgi
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse as urlparse
import io

import mmcv
from mmdet.apis import inference_detector, show_result_pyplot
import numpy as np
import torch
import base64
from PIL import Image 
import json
 
PATH = './weights/'
device = torch.device("cuda")
model = torch.load(PATH + 'model.pt')
model.to(device)
print("setup")
with open("./item.json", 'rt', encoding="UTF-8") as f:
    items = json.loads(f.read())

class myHandler(BaseHTTPRequestHandler):
  def do_POST(self):
    self.send_response(200)
    self.send_header('Content-type', 'image/jpeg')
    self.end_headers()

    ctype, pdict = cgi.parse_header(
        self.headers['content-type'])

    pdict['boundary'] = bytes(pdict['boundary'], "utf-8")

    if ctype == 'multipart/form-data':
        fields = cgi.parse_multipart(self.rfile, pdict)
        messagecontent = fields.get('image')
        # print(fields)
        print(type(messagecontent))
        img = Image.open(io.BytesIO(messagecontent[0]))
        img.save("./test.jpg")
        img = mmcv.imread('./test.jpg')
        result = inference_detector(model, img)
        ret = {}
        for (i, res) in zip(range(len(result)), result):
            if len(res) > 0:
                for arr in res:
                    if arr[-1] > 0.4:
                      if items[model.CLASSES[i]] not in ret.keys():
                          ret[items[model.CLASSES[i]]] = 1
                      else:
                          ret[items[model.CLASSES[i]]] += 1
                    print(items[model.CLASSES[i]], end=": ")
                    print(arr[-1])
        print(ret)
        body = f'{ret}'
        self.wfile.write(body.encode())
 
httpd = HTTPServer(('', 8004), myHandler)
httpd.serve_forever()