#!/usr/bin/env python3

import cv2
import depthai as dai
import serial
import struct
import time
from threading import Thread

# =========================
# RS485 CONFIGURATION
# =========================

PORT = "/dev/ttyTHS1"

BAUDRATE = 9600
BYTESIZE = 8
PARITY = 'N'
STOPBITS = 1
TIMEOUT = 1

# =========================
# SERIAL SETUP
# =========================

ser = serial.Serial(
    port=PORT,
    baudrate=BAUDRATE,
    bytesize=BYTESIZE,
    parity=PARITY,
    stopbits=STOPBITS,
    timeout=TIMEOUT
)

print(f"Connected to {PORT}")

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
# SEND FUNCTION
# =========================

def send_message(message: str):
    data = message.encode("utf-8")
    packet = encode_packet(data)

    ser.write(packet)

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
        
        err_px = tagsCX[0] - (frame.shape[1] / 2.0)
        
        # send data with rs485 from jetson nano to vex brain
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
