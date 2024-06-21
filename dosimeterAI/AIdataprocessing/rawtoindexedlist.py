import os
import re

edepdir = '/home/uakgun/src/gate-2024/output/eDep'
pedir = '/home/uakgun/src/gate-2024/output/Particleenergy(MeV)'
photondir = '/home/uakgun/src/gate-2024/output/photoncount'
btdir = '/home/uakgun/src/gate-2024/output/beamtime'

open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/eDep.txt', 'w').close()
open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/penergy.txt', 'w').close()
open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/PhotonCount.txt', 'w').close()
open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/BeamTime.txt', 'w').close()

# iterate over files in directory
for filename in os.listdir(edepdir):
	f = os.path.join(edepdir, filename)
	rd = open(f)
	for filename in os.scandir(edepdir):
		if filename.is_file():
			print(filename.path)

	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/eDep.txt', 'a')
	edepcontent = rd.readlines()

	q.write(edepcontent[6])

q = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/penergy.txt', 'a')

for filename in os.listdir(pedir):
    res = re.findall("AIsim-(\d+).txt", filename)
    if not res: continue
    q.write(res[0] + '\n') # You can append the result to a list

for filename in os.listdir(photondir):
	f = os.path.join(photondir, filename)
	rd = open(f)
	for filename in os.scandir(edepdir):
		if filename.is_file():
			print(filename.path)
	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/PhotonCount.txt', 'a')
	phocontent = rd.readlines()

	q.write(phocontent[0]+ '\n')

for filename in os.listdir(btdir):
	f = os.path.join(btdir, filename)
	rd = open(f)
	for filename in os.scandir(btdir):
		if filename.is_file():
			print(filename.path)

	q = open('/home/uakgun/src/gate-2024/AIdataprocessing/rawtoindexdata/BeamTime.txt', 'a')
	btcontent = rd.readlines()

	q.write(btcontent[0] + '\n')

print('done')