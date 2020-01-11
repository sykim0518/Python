import numpy as np
from matplotlib import pyplot as plt

Freq=60
F_s=441000
Amp=1

t=np.linspace(0,1,F_s)
x=Amp*np.sin(2*np.pi*Freq*t)

plt.plot(t,x)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()