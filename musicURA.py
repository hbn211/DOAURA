"""
Author MATLAB/MUSIC Development: Pedro Lemos
Author Python: Heriberto Brill Nonemacher
Date: March 2022
Contact: hbn211@gmail.com
|Python 3.8.10 64-bit|scipy==1.8.0|numpy==1.21.5|matplotlib==3.5.1|
"""
from cmath import pi,sin,cos,exp
from matplotlib.pyplot import tight_layout
from numpy import append,gradient,matrix,matmul,sqrt,squeeze,array,kron,zeros,absolute,max,argmax,argsort,arange
from scipy.linalg import eig

def musicURA(x,**kwargs):
    """
    % ========= MUSIC ALGORITHM ========= %\\
    Author MATLAB: Pedro Lemos\\
    Author PYTHON: Heriberto & Pedro\\
    License:\\
    Function to calc URA 16\\
    x is a matrix data (Antennas(complex) |ROWS| x time |COLS|) \\
    optional\\
        'd='            # Distance between elements in Wavelenght. \n\t\tDefault: 0.32 Wavelength\\
        'M='            # Number of incident signals.              \n\t\tDefault: 1\\
        'factordeg='    # Define the degree resolution.            \n\t\tDefault: 1 deg\\
        'verbose='      # Print some infos                         \n\t\tDefault: False\\
        'savepicture='  # Save the QMusic                          \n\t\tDefault: False\\
     #>>> C is not working yet <<<#
    """   
    L = x.shape[1] # N samples
    N = x.shape[0] # N antennas

    if ((N % 2) != 0):
        print("X elements must to have the same Y elements.")
        from sys import exit
        exit()

    C=kwargs.get('C') # Distance between elements in Wavelenght
    if C is None:
        pass
    else:
        if(array(C).shape[0] != N and array(C).shape[1] != N):
            print('C invalid')
            pass
        else:
            C = matrix(squeeze(C))

    d=kwargs.get('d') # Distance between elements in Wavelenght
    if(d == None):
        d=0.32       # default
    else:
        if (d == 0):
            print('Value',d,'not allowed for d.\n-> Try non-zero and positive value')
            from sys import exit
            exit()
   

    M=kwargs.get('M') # Number of incident signals
    if(M == None):
         M=1          # default
    else:
        if (M < 1 or M > N):
            print('Value',M,'must be between 1 and',N)
            from sys import exit
            exit()

    factordeg=kwargs.get('factordeg') #  1degree / factordeg
    if(factordeg == None):
         factordeg=1          # default
    else:
        if(type(factordeg) != int):
            print("factordeg must be integer.")
            from sys import exit
            exit()

    verbose=kwargs.get('verbose')
    if (verbose == True):
        print("d:",d)
        print("M:",M)
        print("factordeg:",factordeg)

    Rxx = (matmul(matrix(x),matrix(x).H))/L #

    D,Q = eig(Rxx)

    I = argsort(D)[::-1] #Find r largest eigenvalues     

    Q=Q[:,I] #Sort the eigenvectors to put signal eigenvectors first

    Qn=Q[:,M:N] #Get the noise eigenvectors        

    azimuth = arange(-180,180+1/factordeg,1/factordeg)

    azimuth_rad=array(azimuth)*pi/180
 
    elevation = arange(0,90+1/factordeg,1/factordeg) #Including the endpoint 
  
    elevation_rad=array(elevation)*pi/180

    #Steering vector
    a1 = zeros((N,azimuth.shape[0],elevation.shape[0]),dtype=complex)
    
    for i in range(azimuth.shape[0]):
        cosi=2*pi*d*cos(azimuth_rad[i])
        sini=2*pi*d*sin(azimuth_rad[i])
        for j in range(elevation.shape[0]):
            sinEJ = sin(elevation_rad[j])
            phi_x = sinEJ*cosi
            phi_y = sinEJ*sini
            a_x = array([1])
            a_y = array([1])
            for x in range(1,(int(sqrt(N)))):
                a_x = append(a_x,exp(x*1j*phi_x))
                a_y = append(a_y,exp(x*1j*phi_y)) 
            a1[:,i,j] = kron(a_y,a_x)

    #Compute MUSIC �spectrum�
    music_spectrum = zeros((azimuth.shape[0],elevation.shape[0]),dtype=complex)

    qM = matrix(Qn)
    qMqMH = matmul(qM,qM.H)
    # if C is None:
    #     pass
    # else:
    #     CH = matrix(C).H
    #     CHC = matmul(CH,C)
    #     CHqMqMHC = matmul(CH,matmul(qMqMH,C))
    
    for i in range(azimuth.shape[0]):
        for j in range(elevation.shape[0]):
            if C is None:
                aM = matrix(a1[:,i,j]).T
            else:
                aM = matmul(C,matrix(a1[:,i,j]).T)  
            aMH = aM.H

            # if C is None:
            music_spectrum[i,j] =           (aMH*aM) /            \
                                            (aMH*qMqMH*aM)
            # else:
            #     music_spectrum[i,j] =        (aMH*(CHC)*aM) /            \
            #                                 (aMH*CHqMqMHC*aM)
                
    musics = absolute(music_spectrum)
    
    #Find the azimuth and elevation angles corresping to the biggest peak
    Index = argmax(musics)
    # Maxel = max(musics)
    I_row,I_col = ind2sub(musics.shape,Index)

    pict=kwargs.get('savepicture') # Set true to save the Qmusic
    if(pict == True):
        from os.path import dirname,realpath
        from matplotlib.pyplot import figure,imshow,colorbar,clim,savefig,subplots
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        currentdir = dirname(realpath(__file__))
        musiT=musics[:,:].transpose()
        gra=squeeze(gradient(musiT))
        gradmag=sqrt((gra[0,:,:]**2)+(gra[1,:,:]**2))
        figure() 
        fig, (ax1,ax2) = subplots(2,1)
        fig.suptitle('Music',fontsize=20)
        ax1.set_title('Power')
        im1 = ax1.imshow(musiT,cmap='seismic',origin = 'lower',extent =[azimuth.min(), azimuth.max(), elevation.min(), elevation.max()], aspect="auto")
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right",size="5%",pad=0.05)
        colorbar(im1,cax=cax1)
        ax2.set_title('Gradient')
        im2 = ax2.imshow(gradmag,cmap='seismic',origin = 'lower',extent =[azimuth.min(), azimuth.max(), elevation.min(), elevation.max()], aspect="auto")
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right",size="5%",pad=0.05)
        colorbar(im2,cax=cax2)
        tight_layout()
        savefig(currentdir+'\musics.png')
        if (verbose == True): print("Picture saved on",currentdir,'\b\musics.png')

    theta = (I_col / factordeg)
    phi = (I_row / factordeg) - 180

    return theta,phi

def ind2sub(array_shape, ind):
    """
    Function came from MATLAB
    """                         
    rows = (ind.astype("int32") // array_shape[1])
    cols = (ind.astype("int32") % array_shape[1])
    return (rows, cols)
