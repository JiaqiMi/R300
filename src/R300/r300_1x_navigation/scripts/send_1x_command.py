#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

try:
    import serial
except ImportError:
    print("缺少 pyserial，请执行：sudo apt install python3-serial")
    sys.exit(1)


PORT = "/dev/ttyACM0"
BAUDRATE = 460800

# 55 AA 55 AA 5A A5 5A A5 AA 00 00 00 00 00 00 00 00 : 启动发送
CMD = bytes.fromhex(
    "55 AA 55 AA 5A A5 5A A5 AA 00 00 00 00 00 00 00 00"
)

# 55 AA 55 AA 5A A5 5A A5 BB 78 56 34 12 78 56 34 12 ：系统复位
# CMD = bytes.fromhex(
#     "55 AA 55 AA 5A A5 5A A5 BB 78 56 34 12 78 56 34 12"
# )

def main():
    print(f"打开串口: {PORT}")
    print(f"波特率: {BAUDRATE}")
    print("发送报文:", CMD.hex(" ").upper())
    print(f"报文长度: {len(CMD)} bytes")

    try:
        with serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
            write_timeout=1.0,
        ) as ser:
            ser.reset_input_buffer()

            n = ser.write(CMD)
            ser.flush()

            if n != len(CMD):
                print(f"发送失败，仅写入 {n}/{len(CMD)} bytes")
                return

            print("发送成功。")

            # 可选：读取 0.2 秒返回数据
            time.sleep(0.2)
            if ser.in_waiting > 0:
                resp = ser.read(ser.in_waiting)
                print("返回数据:", resp.hex(" ").upper())
            else:
                print("未收到返回数据。")

    except serial.SerialException as e:
        print("串口通信失败：", e)
        print("请检查：")
        print("1. /dev/ttyACM0 是否正确；")
        print("2. one_x_serial_driver 是否已经关闭；")
        print("3. 当前用户是否有 dialout 串口权限。")


if __name__ == "__main__":
    main()
