import argparse
import asyncio
import fractions
import json
import logging
import threading
from multiprocessing import shared_memory
import os
import platform
import ssl

import numpy as np
import cv2


from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame


ROOT = os.path.dirname(__file__)


relay = None
webcam = None
relay2 = None
webcam2 = None

frame_buffer=np.zeros((480,480,3),dtype=np.uint8)

class OpenCVVideoStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, video_source=0):
        global cap
        super().__init__()
        self.frame_counter=0
        # self.cap = cv2.VideoCapture(video_source)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 backend

        # Set the frame width and height
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # # Set the frame rate (optional)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)

        # # Set the pixel format to MJPG
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap=self.cap
    
    async def recv(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture video frame")
        
        frame=cv2.flip(frame,-1)

        pts, time_base = await self.next_timestamp()

        
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts # Frame counter as pts
        video_frame.time_base =time_base  # Assuming 30 fps
        self.frame_counter += 1
        return video_frame

    def close(self):
        self.cap.release()

class OpenCVVideoStreamTrack1(VideoStreamTrack):
    kind = "video"

    def __init__(self, video_source=0):
        global cap
        super().__init__()
        self.frame_counter=0
        # self.cap = cv2.VideoCapture(video_source)
        cap=self.cap
        resolution = (720, 1280)
        crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
        crop_size_h = 270
        resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
        img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
        img_height, img_width = resolution_cropped[:2]  # 450 * 600
        # shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)

        shm_name = "image_shared_memory"
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8,
                                        buffer=existing_shm.buf)
    
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.img_array
        
        # frame=cv2.flip(frame,-1)
        
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts  # Frame counter as pts
        video_frame.time_base = time_base  # Assuming 30 fps
        self.frame_counter += 1
        return video_frame

    def close(self):
        self.cap.release()


class OpenCVVideoStreamTrackGlobalVariable(VideoStreamTrack):
    kind = "video"

    def __init__(self):
        super().__init__()
        self.cnt=0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        # VIDEO_CLOCK_RATE = 90000
        # VIDEO_PTIME = 1 / 30  # 30fps
        # time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        # print(frame_buffer.shape)

        # frame=cv2.flip(frame,-1)
        # self.cnt+=1
        # pts=self.cnt
        video_frame = VideoFrame.from_ndarray(frame_buffer, format="bgr24")
        video_frame.pts = pts  # Frame counter as pts
        video_frame.time_base = time_base  # Assuming 30 fps
        return video_frame



def create_local_tracks(play_from, decode):
    global relay, webcam

    if play_from:
        player = MediaPlayer(play_from, decode=decode)
        return player.audio, player.video
    else:
        if relay is None:
            # webcam = OpenCVVideoStreamTrack1(0)  # 0 for default webcam
            webcam = OpenCVVideoStreamTrackGlobalVariable()  # 0 for default webcam
            relay = MediaRelay()
        return None, relay.subscribe(webcam)

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    print(offer)

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    audio, video = create_local_tracks(
        None, decode=False
    )

    # audio2, video2 = create_local_tracks2()

    if video:
        video_sender = pc.addTrack(video)
        # if args.video_codec:
        #     force_codec(pc, video_sender, args.video_codec)


    # if video2:
    #     pc.addTrack(video2)


    await pc.setRemoteDescription(offer)


    answer = await pc.createAnswer()

    print(answer)
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


pcs = set()


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def main():
    parser = argparse.ArgumentParser(description="WebRTC webcam demo")
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument("--play-from", help="Read the media from a file and sent it.")
    parser.add_argument(
        "--play-without-decoding",
        help=(
            "Read the media without decoding it (experimental). "
            "For now it only works with an MPEGTS container with only H.264 video."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument(
        "--audio-codec", help="Force a specific audio codec (e.g. audio/opus)"
    )
    parser.add_argument(
        "--video-codec", help="Force a specific video codec (e.g. video/H264)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


def main_simple():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, host="0.0.0.0", port=8080)

from time import sleep
def block_function():
    while True:
        print("block_function")
        sleep(1)

from concurrent.futures import ProcessPoolExecutor
def main2():
    thread1=threading.Thread(target=main)
    thread1.start()
    while True:
        print("main")
        sleep(1)

if __name__ == "__main__":
    # main()
    # asyncio.run(main2())
    main_simple()

