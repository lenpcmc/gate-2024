import os

onlyfiles = next(os.walk('/home/vgate/src/gate-2024/output/eDep'))[2] #directory is your directory path as string
ed = open('/home/uakgun/src/gate-2024/AIdataprocessing/eDep.txt', 'r')
pe = open('/home/uakgun/src/gate-2024/AIdataprocessing/penergy.txt', 'r')
pc = open('/home/uakgun/src/gate-2024/AIdataprocessing/PhotonCount.txt', 'r')
bt = open('/home/uakgun/src/gate-2024/AIdataprocessing/BeamTime.txt', 'r')

edcontent = ed.readlines()
pecontent = pe.readlines()
pccontent = pc.readlines()
btcontent = bt.readlines()

q = open('/home/vgate/src/gate-2024/AIdataprocessing/AIdata.csv', 'a')

for i in range(int(onlyfiles)):
	q.write(pecontent[i] + ',' + pccontent[i] + ',' + btcontent[i] + edcontent[i] + '\n')
	
print('done')