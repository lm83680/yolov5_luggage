import asyncio
import websockets
import cv2
import numpy as np

import desktop_v3_detect


async def websocket_handler(websocket, path):
    try:
        async for message in websocket:
            print(f"Received: {message}")
            if message == "start":
                # await desktop_v3_detect.run(websocket= websocket)
                await websocket.send("收到了，图给你")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed with exception: {e}")

async def serve():
    server = await websockets.serve(websocket_handler, 'localhost', 8765)
    return server

async def restart_server(interval):
    server = await serve()
    print("WebSocket server started.")
    try:
        while True:
            await asyncio.sleep(interval)
            server.close()
            await server.wait_closed()
            print("WebSocket server stopped. Restarting now...")
            server = await serve()
            print("WebSocket server restarted.")
    except asyncio.CancelledError:
        server.close()
        await server.wait_closed()
        print("WebSocket server shutting down.")

async def main():
    # 设定重启间隔为600秒，即10分钟
    restart_task = asyncio.create_task(restart_server(600))
    await restart_task




def image_to_bytes(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    # 编码图像为JPEG格式
    _, buffer = cv2.imencode('.jpg', img)
    # 转换为字节流
    img_bytes = buffer.tobytes()
    return img_bytes

if __name__ == '__main__':
    asyncio.run(main())
