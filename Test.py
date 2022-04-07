"""
Author: Heriberto Brill Nonemacher
Date: March 2022
Contact: hbn211@gmail.com
|Python 3.8.10 64-bit|
|pandas==1.4.1|numpy==1.21.5|scikit-rf==0.21.0| -> For Test.py
|scipy==1.8.0|numpy==1.21.5|matplotlib==3.5.1| -> For musicURA16.py
"""
from pandas import read_csv
from numpy import identity, zeros,floor
from time import time
from sys import path
from os.path import dirname,realpath,isfile
from skrf import Network

currentdir = dirname(realpath(__file__))
parentdir = dirname(currentdir)
path.append(parentdir)
from musicURA import musicURA

csvfile = "\Solutions\ForSolve1m0.1T45P45NearFieldCA.csv"

if isfile(currentdir+csvfile):

    data = read_csv(currentdir+csvfile)
    data = data.drop(data.columns[[0]], axis=1)

    numdata=data.to_numpy(dtype=float)
    num_samples = data.shape[1]
    num_antennas = int(data.shape[0]/2)

    x_iq = zeros((num_antennas,num_samples),dtype=complex)
    for sample in range(num_samples):               #columns
        for antenna in range(num_antennas):         #rows
            i = numdata[2*antenna,sample]           #I
            q = numdata[2*antenna+1,sample]         #Q
            x_iq[antenna,sample] = i+1j*q           #I+jQ

    if(isfile(currentdir+"\\S.s"+str(num_antennas)+"p")):
        print("S parameters available for mutual coupling...\nWould you like to use? (y or n):")
        useS = input()
        if useS == "y":
            NormalizationImpedance = 50
            ring_slot = Network(currentdir+"\\S.s"+str(num_antennas)+"p")
            Sparam = ring_slot.s
            if(ring_slot.f.shape[0] > 1):
                print("Available frequencies:\n")
                for fi in range(ring_slot.f.shape[0]):     
                    print(fi+1,"->\t",ring_slot.f[fi]/1000000,"Mhz")

                print("Select the frequency position (1 to ",str(ring_slot.f.shape[0]),"):")
                sel = input("> ")
                if sel.isnumeric():
                    if(int(sel) < 1 or int(sel) > ring_slot.f.shape[0]):
                        sel = floor(ring_slot.f.shape[0]/2)
                        print("This answer is out of range...\nSelected in middle range in the position",int(sel),"with the frequency:",ring_slot.f[int(sel)-1]/1000000,"Mhz")
                    else:
                        print("Selected",int(sel),"with the frequency:",ring_slot.f[int(sel)-1]/1000000,"Mhz")
                else:
                    print("This is not a integer number!")
                    sel = floor(ring_slot.f.shape[0]/2)
                    print("Selected in middle range in the position",int(sel),"with the frequency:",ring_slot.f[int(sel)-1]/1000000,"Mhz")
            else:
                sel = 1

            Cm = identity(num_antennas) - Sparam[(int(sel)-1),:,:]

    else:
        Cm = None

    #DO THE MAGIC
    print("Solving...")
    start_time = time()
    theta,phi = musicURA(x_iq,d=0.3255919,factordeg=1,savepicture=True,C=Cm)#
    print("--- %s seconds ---" % (time() - start_time))
    
    print("Phi:",phi) #az
    print("Theta:",theta) #el

else:
    print("Where is the HFSS file?")
