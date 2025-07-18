import pyfirmata
comport = 'COM4'
board = pyfirmata.Arduino(comport)
pin1 = board.get_pin('d:13:o')
pin2 = board.get_pin('d:12:o')
pin3 = board.get_pin('d:8:o')
pin4 = board.get_pin('d:7:o')
def out(message):
    if message == 'A':
        pin3.write(1)
        pin1.write(0)
        pin4.write(0)
        pin2.write(0)
    if message == 'D':
        pin3.write(0)
        pin1.write(0)
        pin4.write(0)
        pin2.write(1)
    if message == 'Al':
        pin1.write(1)
        pin4.write(1)
        pin2.write(0)
        pin3.write(0)
    