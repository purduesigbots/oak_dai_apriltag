#!/usr/bin/env python3

import os
import cv2
import depthai as dai
import serial
import serial.tools.list_ports
import struct
import time
from threading import Thread

# =========================
# USB SERIAL (V5 user port)
# =========================
# Plug Jetson USB ↔ brain micro-USB.
# Brain shows two CDC devices on Linux:
#   /dev/ttyACM0  = system (uploads / PROS protocol)  ← do NOT use for app data
#   /dev/ttyACM1  = user   (program stdin/stdout)     ← use this one
# Override with:  V5_PORT=/dev/ttyACM1 python3 apriltag_xalign.py

BAUDRATE = 115200
BYTESIZE = 8
PARITY = "N"
STOPBITS = 1
TIMEOUT = 1


def find_v5_user_port() -> str:
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
            "No V5 USB serial ports found. Connect micro-USB Jetson↔brain, then retry."
        )
    if len(ports) >= 2:
        # Second ACM device is usually the user port on Linux
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
# =========================
# PACKET FORMAT
# =========================
#
# Packet Structure:
#
# [START][LENGTH][DATA][CHECKSUM]
#
# START     = 0xAA
# LENGTH    = Number of bytes in DATA
# DATA      = Payload bytes
# CHECKSUM  = Sum(DATA) % 256
#
# Example:
# AA 05 48 45 4C 4C 4F 74
#
# =========================


START_BYTE = 0xAA


def calculate_checksum(data_bytes):
    return sum(data_bytes) % 256


def encode_packet(data: bytes) -> bytes:
    length = len(data)
    checksum = calculate_checksum(data)

    packet = struct.pack("BB", START_BYTE, length)
    packet += data
    packet += struct.pack("B", checksum)

    return packet


def decode_packet(packet: bytes):
    if len(packet) < 3:
        return None

    start = packet[0]

    if start != START_BYTE:
        return None

    length = packet[1]

    data = packet[2:2 + length]

    received_checksum = packet[2 + length]

    calculated_checksum = calculate_checksum(data)

    if received_checksum != calculated_checksum:
        print("Checksum failed")
        return None

    return data

# =========================
# SEND FUNCTION - tagsCX[0]
# =========================

def send_message(message: str):
    data = message.encode("utf-8")
    packet = encode_packet(data)

    ser.write(packet)
    ser.flush()

    print(f"Sent: {packet.hex(' ')}")

# =========================
# RECEIVE FUNCTION
# =========================

def receive_loop():
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

            full_packet = (
                start +
                length_bytes +
                data +
                checksum
            )

            decoded = decode_packet(full_packet)

            if decoded:
                print("Received:", decoded.decode("utf-8"))

        except Exception as e:
            print("Receive error:", e)

# =========================
# START RECEIVER THREAD
# =========================

receiver = Thread(target=receive_loop, daemon=True)
receiver.start()

with dai.Pipeline() as pipeline:
    hostCamera = pipeline.create(dai.node.Camera).build()
    aprilTagNode = pipeline.create(dai.node.AprilTag)
    aprilTagNode.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_CIR21H7)
    hostCamera.requestOutput((1920, 1080)).link(aprilTagNode.inputImage)
    passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
    outQueue = aprilTagNode.out.createOutputQueue()

    color = (0, 255, 0)
    startTime = time.monotonic()
    counter = 0
    fps = 0.0
    
    pipeline.start()
    while pipeline.isRunning():
        aprilTagMessage = outQueue.get()
        assert(isinstance(aprilTagMessage, dai.AprilTags))
        aprilTags = aprilTagMessage.aprilTags

        counter += 1
        currentTime = time.monotonic()
        if (currentTime - startTime) > 1:
            fps = counter / (currentTime - startTime)
            counter = 0
            startTime = currentTime

        passthroughImage: dai.ImgFrame = passthroughOutputQueue.get()
        frame = passthroughImage.getCvFrame()

        def to_int(tag):
            return (int(tag.x), int(tag.y))

        tagsCX = []

        for tag in aprilTags:
            topLeft = to_int(tag.topLeft)
            topRight = to_int(tag.topRight)
            bottomRight = to_int(tag.bottomRight)
            bottomLeft = to_int(tag.bottomLeft)

            centerX = int((topLeft[0] + bottomRight[0]) / 2)
            tagsCX.append(centerX)

            # center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))

            cv2.line(frame, topLeft, topRight, color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, topRight,bottomRight, color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, bottomRight,bottomLeft, color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, bottomLeft,topLeft, color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(tag.id)
            # cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.putText(frame, f"fps: {fps:.1f}", (200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        # cv2.imshow("detections", frame)
        
        if len(tagsCX) != 1:
            print("bad")
        
        # err_px = tagsCX[0] - (frame.shape[0] / 2.0)
        
        # send data over USB user serial to V5 stdin
        # =========================
        # MAIN LOOP
        # =========================

        msg = input("Enter message: ")

        if msg.lower() == "exit":
            break

        send_message(msg)
        
        if cv2.waitKey(1) == ord("q"):
            break
        
    ser.close()
