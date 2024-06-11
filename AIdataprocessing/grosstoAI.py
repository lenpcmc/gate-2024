import os

edepdir = '/home/uakgun/src/gate-2024/output/eDep'
pedir = '/home/uakgun/src/gate-2024/output/Particleenergy(MeV)'
raddir = '/home/uakgun/src/gate-2024/output/beamrad'
photondir = '/home/uakgun/src/gate-2024/output/photoncount'
btdir = '/home/uakgun/src/gate-2024/output/beamtime'

# iterate over files in directory
for filename in os.listdir(edepdir):
	f = os.path.join(edepdir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/eDepdata.txt', 'a')
	edepcontent = rd.readlines()

	q.write(edepcontent[6] + ',')

for filename in os.listdir(pedir):
	f = os.path.join(pedir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/Particleenergy(MeV).txt')
	pecontent = rd.readlines()

	q.write(pecontent + ',')

for filename in os.listdir(raddir):
	f = os.path.join(raddir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/BeamRadius.txt')
	radcontent = rd.readlines()

	q.write(radcontent + ',')

for filename in os.listdir(photondir):
	f = os.path.join(raddir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/PhotonCount.txt')
	phocontent = rd.readlines()

	q.write(phocontent + '.txt')

for filename in os.listdir(btdir):
	f = os.path.join(raddir, filename)
	rd = open(filename)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/BeamTime.txt')
	timecontent = rd.readlines()

	q.write(timecontent + ',')