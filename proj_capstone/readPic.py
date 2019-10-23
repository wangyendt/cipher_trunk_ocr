import serial #导入串口模块
import time
from imgproc.cipher_trunk_ocr import *

def takePhoto(file_name=None):
    '''
    :param file_name: 图片保存地址
    :return: 返回一个元组，(code,msg）
              code = 1 运行正常 msg 为保存文件地址
              code = 0 运行异常 msg 为错误信息
    '''
    if file_name is None:
        file_name = 'img/1.jpg'
    try:
        portx = "COM3"
        bps = 115200
        # 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
        timex = None
        ser = serial.Serial(portx, bps, timeout=timex)
        ser.bytesize = serial.EIGHTBITS  # 8位数据位
        ser.parity = serial.PARITY_EVEN  # 偶校验

        if not ser.is_open:
            return (0,'串口打开失败')
        # print("串口详情参数：", ser)

        # 十六进制命令
        cmd_str = bytes().fromhex('55 00 00 00 00 13 FF FF FF FF F5')

        for i in range(0, 10): #尝试发送10次拍照命令
            result = ser.write(cmd_str) # 发送命令

            str = ser.read(ser.in_waiting)

            if len(str) > 0: #如果接收到回报消息则停止

                pic_beg = str.find(bytes.fromhex('FF D8'))
                pic_end = str.find(bytes.fromhex('FF D9'))

                pic_str = str[pic_beg:pic_end]
                with open(file_name, 'wb') as wf:
                    wf.write(pic_str)
                break
            time.sleep(1)

        if len(str) > 0:
            return (1 , file_name)
        else:
            return (0 , '未获取到照片信息')
        ser.close()  # 关闭串口

    except Exception as e:
        print("---异常---：", e)
        return (0,string(e))



if __name__=="__main__":
    model_path = r'imgproc/model_digit_cpu.mdl'
    classifier = Classifier(model_path)
    for i in range(1,4): # 连续拍9 张照片
        path = f'img/{i}.jpg'
        code,msg = takePhoto(path)
        if code > 0 :
            print(f'图片保存=>   {msg}')
            print(classifier.fetch_digits_from_image(path))
        else :
            print(f'异常=>   {msg}')