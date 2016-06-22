
K = mle[13]
m1 = 1.31 * 1.9891e+30
ecc = sqrt(mle[11]**2 + mle[12]**2)
esinw = mle[12]
ars = pp.get_ar(mle[0],mle[8])
b = mle[9]
i = arccos((b/ars) * ((1+esinw)/(1-ecc**2)))
G = 6.67E-11
P = mle[8] * 86400
m2 = K * m1**(2./3.) * sqrt(1-ecc**2) * ((2*pi*G)/P)**(-1./3.)
mjup = 1.8986E27



K = g2[:,13]
m1 = normal(1.31*msun,0.10*msun,size=len(g2[:,0]))
ecc = sqrt(g2[:,11]**2 + g2[:,12]**2)
esinw = g2[:,12]
ars = pp.get_ar(g2[:,0],g2[:,8])
b = g2[:,9]

i = arccos((b/ars) * ((1+esinw)/(1-ecc**2)))
P = g2[:,8] * 86400
m2 = K * m1**(2./3.) * sqrt(1-ecc**2) * ((2*pi*G)/P)**(-1./3.)
mjup = 1.8986E27


