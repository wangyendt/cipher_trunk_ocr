# 串口查询

# import serial.tools.list_ports
# port_list = list(serial.tools.list_ports.comports())
# if len(port_list) == 0:
#    print('找不到串口')
# else:
#     for i in range(0,len(port_list)):
#         print(port_list[i])
import time
import serial #导入模块
try:
    portx="COM3"
    bps=115200
    #超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
    timex=None
    ser=serial.Serial(portx,bps,timeout=timex)
    ser.bytesize = serial.EIGHTBITS  # 8位数据位
    ser.parity = serial.PARITY_EVEN  # 偶校验
    print("串口详情参数：", ser)


    #十六进制的发送
    cmd_str = bytes().fromhex('55 00 00 00 00 13 FF FF FF FF F5')

    str = ser.read(ser.in_waiting)
    for i in range(0,10):
        result = ser.write(cmd_str)
        print("写总字节数:", result)
        str = ser.read(ser.in_waiting)

        if len(str) > 0:
            print(str)
            pic_beg = str.find(bytes.fromhex('FF D8'))
            pic_end = str.find(bytes.fromhex('FF D9'))

            print(pic_beg,pic_end)
            pic_str = str[pic_beg:pic_end]
            print(pic_str)
            with open(f'img/{i}.jpg', 'wb') as wf:

                wf.write(pic_str)
        time.sleep(1)

  #十六进制的读取
  # print(ser.read())#读一个字节
  # while True:
  #     if ser.in_waiting:
  #       str = ser.read(ser.in_waiting)
  #       if (str == "exit"):  # 退出标志
  #           break
  #       else:
  #           print("收到数据：", str)

    print("---------------")
    ser.close()#关闭串口

except Exception as e:
    print("---异常---：",e)