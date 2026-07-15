#!/usr/bin/env python3

import os
import cv2
import depthai as dai
import serial
import serial.tools.list_ports
import struct
import time
from threading import Thread

# usb link to the v5 user port (usually ttyacm1)
# override with: V5_PORT=/dev/ttyACM1 python3 apriltag_xalign.py

APRILTAG_VISUAL_DEBUG = True  # set false to skip frame pulls / overlays

BAUDRATE = 115200
BYTESIZE = 8
PARITY = "N"
STOPBITS = 1
TIMEOUT = 1

START_BYTE = 0xAA  # packet: aa | len | data... | checksum


def find_v5_user_port() -> str:
    # pick the brain's user cdc port
    override = os.environ.get("V5_PORT")
    if override:
        return override

    ports = [
        p.device
        for p in serial.tools.list_ports.comports()
        if ("ACM" in p.device) or ("usbmodem" in p.device.lower())
    ]
    ports = sorted(ports)
    if not ports:
        raise RuntimeError(
            "No V5 USB serial ports found. Connect Jetson USBA to Brain microUSB, then retry."
        )
    # second acm device is usually the user port on linux, shouldn't be falling
    # back on ttyACM0 unless something is terribly wrong
    if len(ports) >= 2:
        return ports[1]
    return ports[0]


PORT = find_v5_user_port()

ser = serial.Serial(
    port=PORT,
    baudrate=BAUDRATE,
    bytesize=BYTESIZE,
    parity=PARITY,
    stopbits=STOPBITS,
    timeout=TIMEOUT,
)
print(f"Connected to V5 user port {PORT} @ {BAUDRATE}")

def calculate_checksum(data_bytes):
    # cheap integrity check: sum of payload bytes mod 256
    return sum(data_bytes) % 256


def encode_packet(data: bytes) -> bytes:
    # wrap payload as aa | len | data | checksum
    length = len(data)
    checksum = calculate_checksum(data)

    packet = struct.pack("BB", START_BYTE, length)
    packet += data
    packet += struct.pack("B", checksum)

    return packet


def decode_packet(packet: bytes):
    # undo encode_packet; returns None if framing/checksum is bad
    if len(packet) < 3:
        return None

    start = packet[0]
    if start != START_BYTE:
        print("Framing failed: bad start byte!")
        return None

    length = packet[1]
    data = packet[2 : 2 + length]
    received_checksum = packet[2 + length]
    calculated_checksum = calculate_checksum(data)

    if received_checksum != calculated_checksum:
        print("Checksum failed: mismatch!")
        return None

    return data


def send_message(message: str):
    # packetize text and push it to the brain over usb
    data = message.encode("utf-8")
    packet = encode_packet(data)
    ser.write(packet)
    ser.flush()  # make sure it actually hits the wire
    print(f"Sent: {packet.hex(' ')}")


def receive_loop():
    # listen for the brain back communications
    while True:
        try:
            start = ser.read(1)
            if not start:
                continue
            if start[0] != START_BYTE:
                continue

            length_bytes = ser.read(1)
            if not length_bytes:
                continue

            length = length_bytes[0]
            data = ser.read(length)
            checksum = ser.read(1)

            if len(data) != length or len(checksum) != 1:
                continue

            full_packet = start + length_bytes + data + checksum
            decoded = decode_packet(full_packet)
            if decoded:
                print("Received from brain:", decoded.decode("utf-8"))

        except Exception as e:
            print("Receive from brain error:", e)


receiver = Thread(target=receive_loop, daemon=True)
receiver.start()

with dai.Pipeline() as pipeline:
    hostCamera = pipeline.create(dai.node.Camera).build()
    aprilTagNode = pipeline.create(dai.node.AprilTag)
    aprilTagNode.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_CIR21H7)
    hostCamera.requestOutput((1920, 1080)).link(aprilTagNode.inputImage)
    outQueue = aprilTagNode.out.createOutputQueue()

    # only request the camera passthrough stream if we're drawing boxes
    passthroughOutputQueue = None
    if APRILTAG_VISUAL_DEBUG:
        passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()

    color = (0, 255, 0)
    startTime = time.monotonic()
    counter = 0
    fps = 0.0

    pipeline.start()
    while pipeline.isRunning():
        aprilTagMessage = outQueue.get()
        assert isinstance(aprilTagMessage, dai.AprilTags)
        aprilTags = aprilTagMessage.aprilTags

        frame = None
        if APRILTAG_VISUAL_DEBUG:
            # rough fps estimate every second
            counter += 1
            currentTime = time.monotonic()
            if (currentTime - startTime) > 1:
                fps = counter / (currentTime - startTime)
                counter = 0
                startTime = currentTime

            passthroughImage: dai.ImgFrame = passthroughOutputQueue.get()
            frame = passthroughImage.getCvFrame()

        def to_int(pt):
            # depthai corners come back as floats
            return (int(pt.x), int(pt.y))

        tagsCX = []

        for tag in aprilTags:
            topLeft = to_int(tag.topLeft)
            topRight = to_int(tag.topRight)
            bottomRight = to_int(tag.bottomRight)
            bottomLeft = to_int(tag.bottomLeft)

            # use box center x for later alignment math
            centerX = int((topLeft[0] + bottomRight[0]) / 2)
            tagsCX.append(centerX)

            if APRILTAG_VISUAL_DEBUG:
                cv2.line(frame, topLeft, topRight, color, 2, cv2.LINE_AA, 0)
                cv2.line(frame, topRight, bottomRight, color, 2, cv2.LINE_AA, 0)
                cv2.line(frame, bottomRight, bottomLeft, color, 2, cv2.LINE_AA, 0)
                cv2.line(frame, bottomLeft, topLeft, color, 2, cv2.LINE_AA, 0)
                cv2.putText(
                    frame,
                    f"fps: {fps:.1f}",
                    (200, 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    color,
                )

        if len(tagsCX) < 1:
            print(f"Expected at least 1 tag, but found {len(tagsCX)}")

        # manual tx test over usb while vision loop runs
        msg = input("Enter message: ")
        if msg.lower() == "exit":
            break

        send_message(msg)

        if APRILTAG_VISUAL_DEBUG and cv2.waitKey(1) == ord("q"):
            break

    ser.close()
