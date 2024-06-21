import os

open('/home/uakgun/src/gate-2024/AIdataprocessing/AIdata.csv', 'w').close()

q = open('/home/uakgun/src/gate-2024/AIdataprocessing/AIdata.csv', 'a')

def funct(pe, pc, bt):
	ans = abs(pe* 2 * pc - 5 * bt)
	return ans
	

for i in range(2000):
	pecontent = (i/1000)
	pccontent = (i**2)
	btcontent = (i+35)
	edcontent = funct(pecontent, pccontent, btcontent)
	q.write(str(pecontent) + ',' + str(pccontent) + ',' + str(btcontent) + ',' + str(edcontent) + '\n')
	
print('done')