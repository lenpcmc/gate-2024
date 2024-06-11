x = int(input('(from 0) x coordinate?: '))
y = int(input('(from 0) y coordinate?: '))
z = int(input('(from 0) z coordinate?: '))
ex = int(input('Ex = ?: '))
ey = int(input('Ey = ?: '))
ez = int(input('Ez = ?: '))
bx = int(input('Bx = ?: '))
by = int(input('By = ?: '))
bz = int(input('Bz = ?: '))


f = open('mac/BPefield.txt', 'a')
f.write('#x\ty\tz\tEx\tEy\tEz\tBx\tBy\tBz\n')


def zmake(y2):
   global z
   global y
   global x
   x3 = x
   z3 = z
   for i in range(z3):
           f.write(str(x3) + '\t' + str(y2) + '\t' + str(z3) + '\t' + str(ex) + '\t' + str(ey) + '\t' + str(ez) + '\t' + str(bx) + '\t' + str(by) + '\t' + str(bz) + '\n')
           z3 += -1


def ymake():
    global z
    global y
    global x
    y2 = y
    for i in range(y2):
           zmake(y2)
           y2 += -1


def xmake():
    global z
    global y
    global x
    for i in range(x):
           ymake()
           x += -1


xmake()