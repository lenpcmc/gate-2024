import os

edepdir = '/home/uakgun/src/gate-2024/output/eDep'
pedir = '/home/vgate/src/gate-2024/output/penergy'
photondir = '/home/uakgun/src/gate-2024/output/photoncount'
btdir = '/home/uakgun/src/gate-2024/output/beamtime'

# iterate over files in directory
for filename in os.listdir(edepdir):
	f = os.path.join(edepdir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/eDep.txt', 'a')
	edepcontent = rd.readlines()

	q.write(edepcontent[6] + '\n')

for filename in os.listdir(pedir):
	f = os.path.join(pedir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/penergy.txt', 'a')
	pecontent = rd.readlines()

	q.write(pecontent + '\n')

for filename in os.listdir(photondir):
	f = os.path.join(photondir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/PhotonCount.txt', 'a')
	phocontent = rd.readlines()

	q.write(phocontent + '\n')

for filename in os.listdir(btdir):
	f = os.path.join(btdir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/BeamTime.txt', 'a')
	timecontent = rd.readlines()

	q.write(timecontent + '\n')

print('done')