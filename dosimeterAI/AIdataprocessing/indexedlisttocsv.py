import os

open('/home/uakgun/src/gate-2024/AIdataprocessing/AIdata.csv', 'w').close()

ed = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/eDep.txt', 'r')
pe = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/penergy.txt', 'r')
pc = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/PhotonCount.txt', 'r')
bt = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/BeamTime.txt', 'r')

edcont = ed.readlines()
pecont = pe.readlines()
pccont = pc.readlines()
btcont = bt.readlines()

q = open('/home/uakgun/src/gate-2024/AIdataprocessing/AIdata.csv', 'a')

for i in range(len(edcont)):
	pecontent = pecont[i]
	pccontent = pccont[i]
	btcontent = btcont[i]
	edcontent = edcont[i]
	q.write(pecontent.replace('\n', '') + ',' + pccontent.replace('\n','') + ',' + btcontent.replace('\n', '') + ',' + edcontent.replace('\n', '') + '\n')
	
print('done')