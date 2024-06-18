import socket
import time

#ローアングル照明：チャンネル1、調光値 70 150
#ハイアングル照明：チャンネル2、調光値 220

class LightModule:
    def __init__(self, supply_ip="192.168.4.2", pc_ip="192.168.4.3"):
        self.supply_ip = supply_ip
        self.supply_port = 40001

        self.pc_ip = pc_ip
        self.pc_port = 30001

    def send(self, send_command='@00F000'):
        with  socket.socket(socket.AF_INET,socket.SOCK_STREAM) as TCPsocket:

            #connect
            TCPsocket.connect((self.supply_ip,int(self.supply_port)))
                
            # add check sum
            send_command = send_command + self.make_Checksum(send_command)

            print(send_command)
            

            # convert command to binary
            send_command = send_command + '\r\n'
            send_binary = send_command.encode()

            # send
            TCPsocket.send(send_binary)

            # receive
            rcvdata = TCPsocket.recv(1024)
        
            # trim receive data
            rcvdata = rcvdata.decode()

            # print receive data
            print('--> ', end = "")
            print(rcvdata)

    # チェックサムを作る関数
    # Function to create a checksum
    def make_Checksum(self, command):
        checksum = 0
        for number in range(len(command)):
            checksum += ord(command[number])
        checksum %= 0x100
        return format(checksum,"2X")
    
    def make_dimming_command(self, ch, value):
        command = '@'+ self.convert_ch(ch) + 'F' + str(value).zfill(3)
        return command
    
    def light_on(self, ch):
        command = '@' + self.convert_ch(ch) + 'L1'
        self.send(command)

    def light_off(self, ch):
        command = '@' + self.convert_ch(ch) + 'L0'
        self.send(command)

    def convert_ch(self, ch):
        if isinstance(ch, str):
            if ch == "L1":
                ch = '00'
            elif ch == "L2":
                ch = '01'
        elif isinstance(ch, int):
            ch = str(ch).zfill(2)
        
        return ch

if __name__ == "__main__":
    lm = LightModule()

    # チャンネル1を調光値70に設定
    # send_command = '@00F070'
    send_command = lm.make_dimming_command(0, 70)
    lm.send(send_command)

    # チャンネル1を点灯
    # send_command = '@00L1'
    # lm.send(send_command)
    lm.light_on(ch=0)

    time.sleep(3)

    # チャンネル1を消灯
    # send_command = '@00L0'
    # lm.send(send_command)
    lm.light_off('L1')