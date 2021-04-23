import os
import traceback

import cv2
import json
import uuid
import threading
import threadpool
import logging
import time
from pypylon import pylon
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tornado.web import Application, StaticFileHandler, RequestHandler
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import face_recognition

HTTP_PORT = 8081
root = os.path.dirname(__file__) + "/.."
www_path = root + "/www"

__image_path__ = '../images/'
__out_image_path__ = '../output_images/'

MODEL_STATE = "NONE"  # NONE RUNNING FINISH
MODEL_STAGE = ""  # NONE RUNNING FINISH
MODEL_OUT_FILE = None
DATA = None

pool = threadpool.ThreadPool(5)
executor = ThreadPoolExecutor(5)

logging.basicConfig(
    format='%(asctime)s [%(threadName)s] [%(name)s] [%(levelname)s] %(filename)s[line:%(lineno)d] %(message)s',
    level=logging.INFO
)


def _getNow():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def _makeFileName():
    return 'pylon-' + _getNow() + '.png'


def pylon_cam_shot():
    tlf = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera(tlf.CreateFirstDevice())
    camera.Open()
    # camera.RegisterConfiguration()
    camera.ExposureTimeRaw = 1000000
    camera.StartGrabbing()
    image = pylon.PylonImage()
    output = _makeFileName()
    with camera.RetrieveResult(2000) as result:
        image.AttachGrabResultBuffer(result)
        image.Save(pylon.ImageFileFormat_Png, __image_path__ + output)
        image.Release()
    camera.StopGrabbing()
    camera.Close()
    return output


def v_2gray(_im):
    return cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)


def v_pool(_im, ksize):
    r = np.zeros((int(_im.shape[0] / ksize), int(_im.shape[1] / ksize)), dtype=np.uint8)
    for h in range(0, int(_im.shape[0] / ksize)):
        for w in range(0, int(_im.shape[1] / ksize)):
            cp = []
            for j in range(0, ksize):
                for i in range(0, ksize):
                    cp.append(_im[h * ksize + j][w * ksize + i])
            r[h][w] = max(cp)
    return r


def v_equalization(_im):
    return cv2.equalizeHist(_im)


def v_grad(_im, ksize=5):
    return cv2.addWeighted(
        cv2.convertScaleAbs(cv2.Sobel(_im, cv2.CV_16S, 1, 0, ksize=ksize)), 0.5,
        cv2.convertScaleAbs(cv2.Sobel(_im, cv2.CV_16S, 0, 1, ksize=ksize)), 0.5, 0
    )


def v_gauss_blur(_im, ksize=11):
    return cv2.GaussianBlur(_im, (ksize, ksize), 1)


def v_open(_im, ksize=3, loop=1):
    return cv2.dilate(_im, np.ones((ksize, ksize), np.uint8), iterations=loop)


def v_close(_im, ksize=3, loop=1):
    return cv2.erode(_im, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)), iterations=loop)


def v_blackhat(_im, ksize=3):
    return cv2.morphologyEx(_im, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))


def v_tophat(_im, ksize=3):
    return cv2.morphologyEx(_im, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))


def v_threshold(_im, lower, upper):
    _, r = cv2.threshold(_im, lower, upper, cv2.THRESH_TOZERO)
    return r


def v_threshold_inv(_im):
    _, r = cv2.threshold(_im, 254, 256, cv2.THRESH_BINARY_INV)
    return r


def v_binaryzation(_im, lower, upper):
    _, r = cv2.threshold(_im, lower, upper, cv2.THRESH_BINARY)
    return r


def v_save(_im, filename):
    cv2.imwrite(filename, _im)


def v_histogram(_im):
    hist = cv2.calcHist([_im], [0], None, [256], [0, 256])
    _r = {'xAxis': [], 'series': []}
    _i = 0
    for value in hist:
        _r['xAxis'].append(_i)
        _r['series'].append(int(value[0]))
        _i += 1
    global DATA
    DATA = _r
    return _im


"""Advanced algorithm"""


def v_DrawHoughCircles(_raw, _im, min_dist, min_radius, max_radius, bgr=(0, 0, 255), thickness=2):
    _r = _raw.copy()
    _circles = cv2.HoughCircles(_im, cv2.HOUGH_GRADIENT, 1, min_dist, 100, 30, minRadius=min_radius, maxRadius=max_radius)
    _circles = np.uint16(np.around(_circles))
    if len(_circles.shape) != 3:
        return _r
    global DATA
    DATA = []
    for cc in _circles[0, :]:
        cv2.circle(_r, (cc[0], cc[1]), cc[2], bgr, thickness)
        cv2.circle(_r, (cc[0], cc[1]), 2, bgr, thickness)
        DATA.append({'center_x': cc[0], 'center_y': cc[1], 'radius': cc[2]})
    return _r


def v_DrawHoughLines(_raw, _im, rho=4, theta=np.pi / 360, threshold=10, min_line_length=40, bgr=(0, 0, 255), thickness=1):
    _r = _raw.copy()
    _lines = cv2.HoughLinesP(_im, rho, theta, threshold, minLineLength=min_line_length)
    global DATA
    DATA = []
    for [[x1, y1, x2, y2]] in _lines:
        cv2.line(_r, (x1, y1), (x2, y2), bgr, thickness)
        DATA.append({'x0': x1, 'y0': y1, 'x1': x2, 'y1': y2})
    return _r


def v_DrawMinRect(_raw, _im, bgr=(0, 0, 255), thickness=1):
    _r = _raw.copy()
    _lines = v_hough_lines(_im)
    _points = flatMap(_lines)
    _rect = cv2.minAreaRect(np.array(_points))
    _box = cv2.boxPoints(_rect)
    _box = np.int0(_box)
    cv2.drawContours(_r, [_box], 0, bgr, thickness)
    return _r


def v_DrawKeyPoints(_raw, _im, gap_points=3, bgr=(0, 0, 255), thickness=2):
    _r = _raw.copy()
    _points = v_key_points(_im, gap_points)
    for _point in _points:
        cv2.circle(_r, (_point[0], _point[1]), int(thickness/2), bgr, thickness=thickness)
    return _r


def v_DrawEdge(_raw, _im, bgr=(0, 0, 255), thickness=2):
    _r = _raw.copy()
    contours, hierarchy = cv2.findContours(_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(_r, contours, -1, bgr, thickness)
    return _r


def area(ar):
    return abs(ar[1][0] - ar[0][0]) * abs(ar[1][1] - ar[2][1])


def v_DrawTargetArea(_raw, _im, min_rect_size=50, bgr=(0, 0, 255), thickness=2):
    _r = _raw.copy()
    contours, hierarchy = cv2.findContours(_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    global DATA
    DATA = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if int(area(box)) < min_rect_size:
            continue
        DATA.append({'x': x, 'y': y, 'w': w, 'h': h})
        cv2.drawContours(_r, [box], 0, bgr, thickness)
    return _r


def v_MSER(_raw, _im, min_area=25, max_area=1000, bgr=(0, 0, 255), thickness=2):
    _r = _raw.copy()
    _MSER = cv2.MSER_create(_min_area=min_area, _max_area=max_area)
    regions, boxes = _MSER.detectRegions(_im)
    global DATA
    DATA = []
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(_r, (x, y), (x + w, y + h), bgr, thickness)
        DATA.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
    return _r


"""Key points extraction"""


def v_key_points(_im, gap_point=3):
    key_points = []
    contours, hierarchy = cv2.findContours(_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for line in contours:
        count = 0
        for [[x, y]] in line:
            if count % gap_point == 0:
                key_points.append([x, y])
            count += 1
    return key_points


def v_hough_lines(_im):
    r_lines = []
    lines = cv2.HoughLinesP(_im, 4, np.pi / 360, 10, minLineLength=40)
    for [[x1, y1, x2, y2]] in lines:
        r_lines.append([[x1, y1], [x2, y2]])
    return r_lines


def flatMap(_p):
    _r = []
    for _j in _p:
        for _i in _j:
            _r.append(_i)
    return _r


"""--------------------------"""


def v_face_detection(_im, bgr=(0, 0, 255)):
    _r = _im.copy()
    face_locations = face_recognition.face_locations(_im)
    for (A, B, C, D) in face_locations:
        cv2.rectangle(_r, (D, A), (B, C), bgr, 2)
    return _r


def v_face_recognition(_im, person_image, person_id, bgr=(0, 0, 255), thickness=2):
    _r = _im.copy()
    _person_encoding = face_recognition.face_encodings(face_recognition.load_image_file(__image_path__ + '/' + person_image))[0]
    face_locations = face_recognition.face_locations(_im)
    logging.info("Finding %d faces!" % len(face_locations))
    for (A, B, C, D) in face_locations:
        cv2.rectangle(_r, (D, A), (B, C), bgr, thickness)
        _extract_face = _im[A:C, D:B]
        _extract_face_encoding = face_recognition.face_encodings(_extract_face)
        # print(_person_encoding.shape)
        # print(len(_extract_face_encoding))
        if len(_extract_face_encoding) == 0:
            continue
            # print("extract face shape: " + str(_extract_face.shape))
            # cv2.imshow("haha", _extract_face)
            # cv2.waitKey()
        r = face_recognition.compare_faces(_person_encoding, _extract_face_encoding)
        # print(r)
        if r[0]:
            cv2.putText(_r, person_id, (D, A), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, thickness)
    return _r


"""--------------------------"""


def M_parse(_js: str):
    try:
        global DATA
        DATA = None
        _js = json.loads(_js)
        logging.info("request: " + json.dumps(_js))
        _im = cv2.imread(__image_path__ + _js['image'])
        _p_im = _im
        for _mo in _js['mo']:
            logging.info("Now, %s!" % _mo['proc'])
            global MODEL_STAGE
            MODEL_STAGE = _mo['proc']
            if _mo['proc'] == 'MakeGray':
                _p_im = v_2gray(_p_im)
            elif _mo['proc'] == 'Histogram':
                _ = v_histogram(_p_im)
            elif _mo['proc'] == 'Inv':
                _p_im = v_grad(_p_im)
            elif _mo['proc'] == 'Grad':
                _p_im = v_grad(_p_im, _mo['ksize'])
            elif _mo['proc'] == 'Equalization':
                _p_im = v_equalization(_p_im)
            elif _mo['proc'] == 'Pool':
                _p_im = v_tophat(_p_im, _mo['ksize'])
            elif _mo['proc'] == 'TopHat':
                _p_im = v_tophat(_p_im, _mo['ksize'])
            elif _mo['proc'] == 'BlackHat':
                _p_im = v_blackhat(_p_im, _mo['ksize'])
            elif _mo['proc'] == 'Open':
                _p_im = v_open(_p_im, _mo['ksize'], _mo['loop'])
            elif _mo['proc'] == 'Close':
                _p_im = v_close(_p_im, _mo['ksize'], _mo['loop'])
            elif _mo['proc'] == 'Threshold':
                _p_im = v_threshold(_p_im, _mo['lower'], _mo['upper'])
            elif _mo['proc'] == 'Binaryzation':
                _p_im = v_binaryzation(_p_im, _mo['lower'], _mo['upper'])
            elif _mo['proc'] == 'GaussBlur':
                _p_im = v_gauss_blur(_p_im)
            elif _mo['proc'] == 'DrawHoughCircles':
                _p_im = v_DrawHoughCircles(_im, _p_im, _mo['Dmin'], _mo['Rmin'], _mo['Rmax'], thickness=_mo['thick'])
            elif _mo['proc'] == 'DrawHoughLines':
                _p_im = v_DrawHoughLines(_im, _p_im, thickness=_mo['thick'])
            elif _mo['proc'] == 'DrawMinRect':
                _p_im = v_DrawMinRect(_im, _p_im, thickness=_mo['thick'])
            elif _mo['proc'] == 'DrawKeyPoints':
                _p_im = v_DrawKeyPoints(_im, _p_im, thickness=_mo['thick'])
            elif _mo['proc'] == 'DrawEdge':
                _p_im = v_DrawEdge(_im, _p_im, thickness=_mo['thick'])
            elif _mo['proc'] == 'DrawTargetArea':
                _p_im = v_DrawTargetArea(_im, _p_im, min_rect_size=_mo['min'], thickness=_mo['thick'])
            elif _mo['proc'] == 'MSER':
                _p_im = v_MSER(_im, _p_im, min_area=_mo['min'], max_area=_mo['max'], thickness=_mo['thick'])
            elif _mo['proc'] == 'FaceDetection':
                _p_im = v_face_detection(_im)
            elif _mo['proc'] == 'FaceRecognition':
                _p_im = v_face_recognition(_im, person_image=_mo['person_image'], person_id=_mo['person_id'], thickness=_mo['thick'])
        output_uuid = str(uuid.uuid1())
        output_uuid.replace('-', '')
        output_uuid = 'p-' + output_uuid + '.png'
        global MODEL_OUT_FILE
        MODEL_OUT_FILE = output_uuid
        cv2.imwrite(__out_image_path__ + output_uuid, _p_im)
        logging.info("cv over! out file: " + output_uuid)
        global MODEL_STATE
        MODEL_STATE = 'FINISH'
        return json.dumps({
            'output': output_uuid
        })
    except Exception as e:
        logging.error(traceback.format_exc())
        MODEL_STATE = 'ERROR'
        return json.dumps({
            'error': str(e)
        })


class PylonCamHandler(RequestHandler):
    def get(self):
        try:
            resp = {
                "image_id": pylon_cam_shot(),
            }
        except Exception as e:
            logging.error(str(e))
            resp = {
                'error': 'Pylon相机未接入，或存在其他问题，请管理员检查系统!'
            }
        logging.info(json.dumps(resp, ensure_ascii=False))
        self.write(resp)


class BaseInfoHandler(RequestHandler):
    def get(self):
        img_ls = os.listdir(www_path + "/pic")
        img_ls.sort()
        resp = {
            "images_url": img_ls,
        }
        logging.info(json.dumps(resp, ensure_ascii=False))
        self.write(resp)


class FileOpsHandler(RequestHandler):
    def post(self):
        req = json.loads(self.request.body.decode(encoding='UTF-8'))
        image_id = req['image']
        ops = req['ops']
        if ops == 'del':
            os.remove(__image_path__ + image_id)
        resp = {'success': True}
        return self.write(resp)


class ModelBuildHandler(RequestHandler):
    def get(self):
        if MODEL_STATE == 'NONE':
            resp = {'state': 'NONE'}
        elif MODEL_STATE == 'RUNNING':
            resp = {'state': 'RUNNING', 'stage': MODEL_STAGE}
        elif MODEL_STATE == 'ERROR':
            resp = {'state': 'ERROR'}
        else:
            resp = {'state': 'FINISH', 'image': MODEL_OUT_FILE}
            if DATA is not None:
                resp.update({'data': DATA})
            print(resp)
        logging.info(json.dumps(resp))
        self.write(resp)

    def post(self):
        # request = json.loads(self.request.body
        # y.decode(encoding='UTF-8'))
        # logging.info(">> type : " + type(self.request.body.decode(encoding='UTF-8')))
        global MODEL_STATE
        MODEL_STATE = 'RUNNING'
        reqs = threadpool.makeRequests(M_parse, [([self.request.body.decode(encoding='UTF-8')], None)])
        [pool.putRequest(req) for req in reqs]
        # executor.submit(M_parse, request)
        # th = threading.Thread(target=M_parse, args=(request, ))
        # th.start()
        logging.info(">> over!")
        self.write({'state': 'RUNNING'})
        # return


def main():
    app = Application([
        (r"/images/info", BaseInfoHandler),
        (r"/model", ModelBuildHandler),
        (r"/pylon", PylonCamHandler),
        (r"/ops", FileOpsHandler),
        (r'^/(.*?)$', StaticFileHandler, {"path": www_path, "default_filename": "index.htm"},),
    ], static_path=www_path)
    server = HTTPServer(app)
    server.listen(HTTP_PORT)
    logging.info("http server start, listen port : {}".format(HTTP_PORT))
    IOLoop.current().start()
    pool.wait()


if __name__ == '__main__':
    main()
