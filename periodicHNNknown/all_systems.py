import numpy as np

class System:
  def __init__(self, f1gen, f2gen, Hgen, spacedim, f1, f2, H, PE, KE, netshape1, netshape2, netshape3, sepshape1, sepshape2, sepshape3, sepshape4, x0, H0, h, LR, periods):
    self.f1gen = f1gen 
    self.f2gen = f2gen 
    self.Hgen = Hgen 
    self.spacedim = spacedim
    self.f1 = f1
    self.f2 = f2
    self.H = H
    self.PE = PE 
    self.KE = KE
    self.netshape1 = netshape1
    self.netshape2 = netshape2
    self.netshape3 = netshape3 
    self.sepshape1 = sepshape1
    self.sepshape2 = sepshape2
    self.sepshape3 = sepshape3
    self.sepshape4 = sepshape4
    self.x0 = x0
    self.H0 = H0
    self.h = h
    self.LR = LR
    self.periods = periods


pendulum = System(lambda x: x[1], lambda x: -np.sin(x[0]), lambda x: 1/2*x[1]**2+(1-np.cos(x[0])), [(-2*np.pi, 2*np.pi), (-1.2, 1.2)], 
			lambda z: z[:,1], lambda z: - np.sin(z[:,0]), lambda x: 1/2*x[:,1]**2+(1-np.cos(x[:,0])),
                        lambda x: (1-np.cos(x[:,0])), lambda x: 1/2*x[:,1]**2,
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [2*np.pi,0.])

# https://www.cfm.brown.edu/people/dobrush/am34/Mathematica/ch3/spherical.html
sphericalpend = System(lambda x: x[1], lambda x: -(((np.sin(x[0])**2) * 1.2)**2/(np.sin(x[0])**3)*np.cos(x[0])-np.sin(x[0])), 
                       lambda x: 0.5*(x[1]**2) + 0.5*(((np.sin(x[0])**2) *1.2)**2)/(np.sin(x[0])**2) + np.cos(x[0])-1, 
                       [(-2*np.pi, 2*np.pi), (-1.2, 1.2)], 
                       lambda z: z[:,1], lambda x: -(((np.sin(x[:,0])**2) * 1.2)**2/(np.sin(x[:,0])**3)*np.cos(x[:,0])-np.sin(x[:,0])),
                       lambda x: 0.5*(x[:,1]**2) + 0.5*(((np.sin(x[:,0])**2) *1.2)**2)/(np.sin(x[:,0])**2) + np.cos(x[:,0])-1,
                       lambda x: (1-np.cos(x[:,0])), lambda x: 1/2*x[:,1]**2,
                       2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [np.pi,0.])

h1 = lambda x: (x[2]*x[3]*np.sin(x[0]-x[1]))/(1+(np.sin(x[0]-x[1])**2))
h2 = lambda x: (x[2]**2+2*x[3]**2-2*x[2]*x[3]*np.cos(x[0]-x[1]))/(2*(1+np.sin(x[0]-x[1])**2)**2)
h1a = lambda x: ((x[:,2]*x[:,3]*np.sin(x[:,0]-x[:,1]))/(1+(np.sin(x[:,0]-x[:,1])**2)))
h2a = lambda x: ((x[:,2]**2+2*x[:,3]**2-2*x[:,2]*x[:,3]*np.cos(x[:,0]-x[:,1]))/(2*(1+np.sin(x[:,0]-x[:,1])**2)**2))


doublepend = System(lambda x: np.asarray([(x[2]-x[3]*np.cos(x[0]-x[1]))/(1+np.sin(x[0]-x[1])**2), (-x[2]*np.cos(x[0]-x[1])+2*x[3])/(1+np.sin(x[0]-x[1])**2)]), 
                    lambda x: np.asarray([-2*np.sin(x[0])-h1(x)+h2(x)*np.sin(2*(x[0]-x[1])), -np.sin(x[1])+h1(x)-h2(x)*np.sin(2*(x[0]-x[1]))]), 
                    lambda x: (x[2]**2+2*x[3]**2-2*x[2]*x[3]*np.cos(x[0]-x[1]))/(2*(1+np.sin(x[0]-x[1])**2))-2*np.cos(x[0])-np.cos(x[1]), 
                    [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
		    lambda x: np.stack([(x[:,2]-x[:,3]*np.cos(x[:,0]-x[:,1]))/(1+np.sin(x[:,0]-x[:,1])**2), (-x[:,2]*np.cos(x[:,0]-x[:,1])+2*x[:,3])/(1+np.sin(x[:,0]-x[:,1])**2)]), 
                    lambda x: np.stack([-2*np.sin(x[:,0])-((x[:,2]*x[:,3]*np.sin(x[:,0]-x[:,1]))/(1+(np.sin(x[:,0]-x[:,1])**2)))+h2a(x)*np.sin(2*(x[:,0]-x[:,1])), 
                    -np.sin(x[:,1])+((x[:,2]*x[:,3]*np.sin(x[:,0]-x[:,1]))/(1+(np.sin(x[:,0]-x[:,1])**2)))-((x[:,2]**2+2*x[:,3]**2-2*x[:,2]*x[:,3]*np.cos(x[:,0]-x[:,1]))/(2*(1+np.sin(x[:,0]-x[:,1])**2)**2))*np.sin(2*(x[:,0]-x[:,1]))]), 
                    lambda x: (x[:,2]**2+2*x[:,3]**2-2*x[:,2]*x[:,3]*np.cos(x[:,0]-x[:,1]))/(2*(1+np.sin(x[:,0]-x[:,1])**2))-2*np.cos(x[:,0])-np.cos(x[:,1])+3,
                    lambda x: x[:,2]**2+2*x[:,3]**2, lambda x: (-2*x[:,2]*x[:,3]*np.cos(x[:,0]-x[:,1]))/(2*(1+np.sin(x[:,0]-x[:,1])**2))-2*np.cos(x[:,0])-np.cos(x[:,0]),
                    4, 32, 1, 2, 22, 22, 1, 0., 0., 0.0001, 0.001, [2*np.pi,2*np.pi,0,0.])


trigo = System(lambda z: - 2 * np.cos(z[1]) * np.sin(z[1]), lambda z: - 2 * np.cos(z[0]) * np.sin(z[0]), lambda z: np.sin(z[0])**2 + np.cos(z[1])**2, [(-2., 2.), (-2., 2.)], 
			lambda z: - 2 * np.cos(z[:,1]) * np.sin(z[:,1]), lambda z: - 2 * np.cos(z[:,0]) * np.sin(z[:,0]), lambda z: np.sin(z[:,0])**2 + np.cos(z[:,1])**2 -1,
                        lambda z: np.sin(z[:,0])**2, lambda z: np.cos(z[:,1])**2-1,
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [np.pi,0.])

arctan = System(lambda z: 2 * z[1] / (z[1]**4 +1), lambda z: - 2 * z[0] / (z[0]**4 +1), lambda z: np.arctan(z[0]**2) + np.arctan(z[1]**2), [(-2., 2.), (-2., 2.)], 
			lambda z: 2 * z[:,1] / (z[:,1]**4 +1), lambda z: - 2 * z[:,0] / (z[:,0]**4 +1), lambda z: np.arctan(z[:,0]**2) + np.arctan(z[:,1]**2),
                        lambda z: np.arctan(z[:,0]**2), lambda z: np.arctan(z[:,1]**2),
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [np.pi,0.])

System2 = System(lambda x: x[1], lambda x: np.cos(x[0]), lambda x: 1/2*x[1]**2+(1-np.sin(x[0])), [(-2*np.pi, 2*np.pi), (-1.2, 1.2)], 
			lambda z: z[:,1], lambda z: np.cos(z[:,0]), lambda x: 1/2*x[:,1]**2+(1-np.sin(x[:,0])) -1,
                        lambda x: (1-np.sin(x[:,0])), lambda x: 1/2*x[:,1]**2,
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [np.pi,0.])

System3 = System(lambda x: x[1], lambda x: 2*np.sin(x[0])*np.cos(x[0]), lambda x: 1/2*x[1]**2+(1-(np.sin(x[0]))**2), [(-2*np.pi, 2*np.pi), (-1.2, 1.2)], 
			lambda z: z[:,1], lambda x: 2*np.sin(x[:,0])*np.cos(x[:,0]), lambda x: 1/2*x[:,1]**2+(1-(np.sin(x[:,0]))**2) -1,
                        lambda x: (1-(np.sin(x[:,0]))**2), lambda x: 1/2*x[:,1]**2,
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [np.pi,0.])

System4 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), lambda x: np.asarray([-np.sin(x[0]+x[1])-1, -np.sin(x[0]+x[1])-1]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-np.cos(x[0]+x[1]))+x[0]+x[1], [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
		    lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), lambda x: np.stack([-np.sin(x[:,0]+x[:,1])-1, -np.sin(x[:,0]+x[:,1])-1]), 
                        lambda x: 1/2*(x[:,2]+x[:,3])**2+(1-np.cos(x[:,0]+x[:,1])+x[:,0]+x[:,1]),
                        lambda x: (1-np.cos(x[:,0]+x[:,1])+x[:,0]+x[:,1]), lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

System5 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), lambda x: np.asarray([np.cos(x[0]+x[1]), np.cos(x[0]+x[1])]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-np.sin(x[0]+x[1])), [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
			lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), lambda x: np.stack([np.cos(x[:,0]+x[:,1]), np.cos(x[:,0]+x[:,1])]), 
                        lambda x:  1/2*(x[:,2]+x[:,3])**2+(1-np.sin(x[:,0]+x[:,1])) -1,
                        lambda x: (1-np.sin(x[:,0]+x[:,1])), lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [2*np.pi,2*np.pi,0.,0.])

System6 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), 
                        lambda x: np.asarray([-2*np.cos(x[0]+x[1])*np.sin(x[0]+x[1])-1, -2*np.cos(x[0]+x[1])*np.sin(x[0]+x[1])-1]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-(np.cos(x[0]+x[1]))**2)+x[0]+x[1], 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
			lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), 
                        lambda x: np.stack([-2*np.cos(x[:,0]+x[:,1])*np.sin(x[:,0]+x[:,1])-1, -2*np.cos(x[:,0]+x[:,1])*np.sin(x[:,0]+x[:,1])-1]),
                        lambda x:  1/2*(x[:,2]+x[:,3])**2+(1-(np.cos(x[:,0]+x[:,1]))**2)+x[:,0]+x[:,1],
                        lambda x: (1-(np.cos(x[:,0]+x[:,1]))**2)+x[:,0]+x[:,1], lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

System7 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), 
                        lambda x: np.asarray([np.cos(x[0]), np.cos(x[1])]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-np.sin(x[0])-np.sin(x[1])), 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
			lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), 
                        lambda x: np.stack([np.cos(x[:,0]), np.cos(x[:,1])]),
                        lambda x:  1/2*(x[:,2]+x[:,3])**2+(1-np.sin(x[:,0])-np.sin(x[:,1])) -1,
                        lambda x: (1-np.sin(x[:,0])-np.sin(x[:,1])), lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

System8 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), 
                        lambda x: np.asarray([-np.sin(x[0])-1, -np.sin(x[1])-1]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-np.cos(x[0])-np.cos(x[1])+x[0]+x[1]), 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
			lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), 
                        lambda x: np.stack([-np.sin(x[:,0])-1, -np.sin(x[:,1])-1]),
                        lambda x:  1/2*(x[:,2]+x[:,3])**2+(1-np.cos(x[:,0])-np.cos(x[:,1])+x[:,0]+x[:,1]) +1,
                        lambda x: (1-np.cos(x[:,0])-np.cos(x[:,1])+x[:,0]+x[:,1]), lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

System9 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), 
                        lambda x: np.asarray([np.cos(x[0]), -np.sin(x[1])]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-np.sin(x[0])-np.cos(x[1])), 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
			lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), 
                        lambda x: np.stack([np.cos(x[:,0]), -np.sin(x[:,1])]),
                        lambda x:  1/2*(x[:,2]+x[:,3])**2+(1-np.sin(x[:,0])-np.cos(x[:,1])),
                        lambda x: (1-np.sin(x[:,0])-np.cos(x[:,1])), lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

System10 = System(lambda x: np.asarray([(x[2]+x[3]), (x[2]+x[3])]), 
                        lambda x: np.asarray([np.cos(x[1])*np.cos(x[0])-1, -np.sin(x[0])*np.sin(x[1])-1]), 
                        lambda x: 1/2*(x[2]+x[3])**2+(1-np.sin(x[0])*np.cos(x[1])+x[0]+x[1]) , 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi), (-1.2, 1.2),(-1.2, 1.2)], 
		    lambda x: np.stack([(x[:,2]+x[:,3]), (x[:,2]+x[:,3])]), 
                        lambda x: np.stack([np.cos(x[:,1])*np.cos(x[:,0])-1, -np.sin(x[:,0])*np.sin(x[:,1])-1]),
                        lambda x:  1/2*(x[:,2]+x[:,3])**2+(1-np.sin(x[:,0])*np.cos(x[:,1])+x[:,0]+x[:,1]) -1 ,
                        lambda x: (1-np.sin(x[:,0])-np.cos(x[:,1]+x[:,0]+x[:,1])), lambda x: 1/2*(x[:,2]+x[:,3])**2,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

System11 = System(lambda z: np.sin(z[1]), lambda z: -np.cos(z[0]), 
                        lambda z: np.sin(z[0]) - np.cos(z[1]), 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi)], 
			lambda z: np.sin(z[:,1]), 
                        lambda z: -np.cos(z[:,0]),
                        lambda z: np.sin(z[:,0]) - np.cos(z[:,1]) +1,
                        lambda z: np.sin(z[:,0]), lambda z: - np.cos(z[:,1]),
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [2*np.pi,2*np.pi])

System12 = System(lambda z: -np.sin(z[0]) * np.sin(z[1])+2, lambda z: -np.cos(z[0]) * np.cos(z[1])-2, 
                        lambda z: np.sin(z[0]) * np.cos(z[1]) + 2*z[0] + 2*z[1], 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi)], 
			lambda z: -np.sin(z[:,0]) * np.sin(z[:,1])+2, 
                        lambda z: -np.cos(z[:,0]) * np.cos(z[:,1])-2,
                        lambda z: np.sin(z[:,0]) * np.cos(z[:,1])+ 2*z[:,0] + 2*z[:,1],
                        lambda z: np.sin(z[:,0])+2*z[:,0], lambda z: - np.cos(z[:,1])+2*z[:,1],
			2, 16, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [2*np.pi,2*np.pi])

System13 = System(lambda z: - 2 * np.cos(z[1]) * np.sin(z[1])+1, lambda z: - 2 * np.cos(z[0]) * np.sin(z[0])-1, 
                        lambda z: np.sin(z[0])**2 + np.cos(z[1])**2+ z[0] + z[1], 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi)], 
			lambda z: - 2 * np.cos(z[:,1]) * np.sin(z[:,1])+1, 
                        lambda z: - 2 * np.cos(z[:,0]) * np.sin(z[:,0])-1,
                        lambda z: np.sin(z[:,0])**2 + np.cos(z[:,1])**2 +z[:,0] +z[:,1],
                        lambda z: np.sin(z[:,0])**2+z[:,0], lambda z: - np.cos(z[:,1])**2+z[:,1],
			2, 16, 1, 1, 11, 11, 1, 0., 1., 0.1, 0.001, [np.pi,np.pi])

System14 = System(lambda z: 2*(np.sin(z[0]) - np.cos(z[1]))*np.sin(z[1])+1, lambda z: -2*(np.sin(z[0]) - np.cos(z[1]))*np.cos(z[0])-1, 
                        lambda z: (np.sin(z[0]) - np.cos(z[1]))**2 + z[0] + z[1], 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi)], 
			lambda z: 2*(np.sin(z[:,0]) - np.cos(z[:,1]))*np.sin(z[:,1])+1, 
                        lambda z: -2*(np.sin(z[:,0]) - np.cos(z[:,1]))*np.cos(z[:,0])-1,
                        lambda z: (np.sin(z[:,0]) - np.cos(z[:,1]))**2 + z[:,1] + z[:,0],
                        lambda z: z[:,1], lambda z: z[:,1] + z[:,0],
			2, 16, 1, 1, 11, 11, 1, 0., 1., 0.1, 0.001, [np.pi,0.])

System15 = System(lambda z: lambda z: np.asarray([2*np.sin(z[10])*np.cos(z[11]), -2*np.sin(z[10])*np.cos(z[11]), 2*np.sin(z[12])*np.cos(z[13]), -2*np.sin(z[12])*np.cos(z[13]),
                            2*np.sin(z[14])*np.cos(z[15]), -2*np.sin(z[14])*np.cos(z[15]), 2*np.sin(z[16])*np.cos(z[17]), -2*np.sin(z[16])*np.cos(z[17]),
                            2*np.sin(z[18])*np.cos(z[19]), -2*np.sin(z[18])*np.cos(z[19])]), 
                        lambda z: lambda z: np.asarray([-2*np.cos(z[0])*np.cos(z[1]), 2*np.sin(z[0])*np.sin(z[1]), -2*np.cos(z[2])*np.cos(z[3]), 2*np.sin(z[2])*np.sin(z[4]),
                            -2*np.cos(z[4])*np.cos(z[5]), 2*np.sin(z[4])*np.sin(z[5]), -2*np.cos(z[6])*np.cos(z[7]), 2*np.sin(z[6])*np.sin(z[7]),
                            -2*np.cos(z[8])*np.cos(z[9]), 2*np.sin(z[8])*np.sin(z[9]),]), 
                        lambda z: (np.sin(z[0])*np.cos(z[1]) + np.sin(z[2])*np.cos(z[3]) + np.sin(z[4])*np.cos(z[5]) + np.sin(z[6])*np.cos(z[7]) + np.sin(z[8])*np.cos(z[9])
                            + np.sin(z[10])*np.cos(z[11]) + np.sin(z[12])*np.cos(z[13]) + np.sin(z[14])*np.cos(z[15]) + np.sin(z[16])*np.cos(z[17]) + np.sin(z[18])*np.cos(z[19])), 
                        [(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),
                         (-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),
                         (-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi),(-2*np.pi, 2*np.pi)], 
			lambda z: np.stack([2*np.sin(z[:,10])*np.cos(z[:,11]), -2*np.sin(z[:,10])*np.cos(z[:,11]), 2*np.sin(z[:,12])*np.cos(z[:,13]), -2*np.sin(z[:,12])*np.cos(z[:,13]),
                            2*np.sin(z[:,14])*np.cos(z[:,15]), -2*np.sin(z[:,14])*np.cos(z[:,15]), 2*np.sin(z[:,16])*np.cos(z[:,17]), -2*np.sin(z[:,16])*np.cos(z[:,17]),
                            2*np.sin(z[:,18])*np.cos(z[:,19]), -2*np.sin(z[:,18])*np.cos(z[:,19])]), 
                        lambda z: np.stack([-2*np.cos(z[:,0])*np.cos(z[:,1]), 2*np.sin(z[:,0])*np.sin(z[:,1]), -2*np.cos(z[:,2])*np.cos(z[:,3]), 2*np.sin(z[:,2])*np.sin(z[:,4]),
                            -2*np.cos(z[:,4])*np.cos(z[:,5]), 2*np.sin(z[:,4])*np.sin(z[:,5]), -2*np.cos(z[:,6])*np.cos(z[:,7]), 2*np.sin(z[:,6])*np.sin(z[:,7]),
                            -2*np.cos(z[:,8])*np.cos(z[:,9]), 2*np.sin(z[:,8])*np.sin(z[:,9]),]),
                        lambda z: (np.sin(z[:,0])*np.cos(z[:,1]) + np.sin(z[:,2])*np.cos(z[:,3]) + np.sin(z[:,4])*np.cos(z[:,5]) + np.sin(z[:,6])*np.cos(z[:,7]) + np.sin(z[:,8])*np.cos(z[:,9])
                             + np.sin(z[:,10])*np.cos(z[:,11]) + np.sin(z[:,12])*np.cos(z[:,13]) + np.sin(z[:,14])*np.cos(z[:,15]) + np.sin(z[:,16])*np.cos(z[:,17]) + np.sin(z[:,18])*np.cos(z[:,19])),
                        lambda z: (np.sin(z[:,0])*np.cos(z[:,1]) + np.sin(z[:,2])*np.cos(z[:,3]) + np.sin(z[:,4])*np.cos(z[:,5]) + np.sin(z[:,6])*np.cos(z[:,7]) + np.sin(z[:,8])*np.cos(z[:,9])), 
                        lambda z: z[:,1] + z[:,0],
			20, 160, 1, 1, 11, 11, 1, 0., 0., 0.1, 0.001, [np.pi,0.])



logarithm = System(lambda x: - 1 + 3/x[1], lambda z: - 1 + 2 /z[0], lambda z: z[0] - np.log(z[0]**2) - z[1] + np.log(z[1]**3), [(1., 3.), (1., 3.)], 
			lambda x: - 1 + 3/x[:,1], lambda z: - 1 + 2 /z[:,0], lambda z: z[:,0] - np.log(z[:,0]**2) - z[:,1] + np.log(z[:,1]**3)+2,
                        lambda z:  z[:,0] - np.log(z[:,0]**2) -1, lambda z: - z[:,1] + np.log(z[:,1]**3) +1,
			2, 16, 1, 1, 11, 11, 1, 1., 2., 0.1, 0.001, [np.pi,0.])

anisotropicoscillator2D = System(
			lambda x: np.asarray([x[2]/np.sqrt(x[2]**2+x[3]**2+1**2), x[3]/np.sqrt(x[2]**2+x[3]**2+1**2)]), 
			lambda x: -np.asarray([1*x[0]+0*x[0]**3,1*x[1]+0.05*x[1]**3]), 
			lambda x: np.sqrt(x[2]**2+x[3]**2+1**2) + 0.5*1*x[0]**2 +0.5*1*x[1]**2 +.25*0*x[0]**4 +.25*0.05*x[1]**4, 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,2]/np.sqrt(x[:,2]**2+x[:,3]**2+1**2), x[:,3]/np.sqrt(x[:,2]**2+x[:,3]**2+1**2)]), 
			lambda x: -np.stack([1*x[:,0]+0*x[:,0]**3,1*x[:,1]+0.05*x[:,1]**3]), 
			lambda x: np.sqrt(x[:,2]**2+x[:,3]**2+1**2) + 0.5*1*x[:,0]**2 +0.5*1*x[:,1]**2 +.25*0*x[:,0]**4 +.25*0.05*x[:,1]**4-1,
                        lambda x: 0.5*1*x[:,0]**2 +0.5*1*x[:,1]**2 +.25*0*x[:,0]**4 +.25*0.05*x[:,1]**4, 
                        lambda x: np.sqrt(x[:,2]**2+x[:,3]**2+1**2) -1,
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.001, 0.01, [np.pi,0.])

henonheiles = System(lambda x: np.asarray([x[2], x[3]]), 
			lambda x: np.asarray([-x[0]-2*1*x[0]*x[1], -x[1]-1*(x[0]*x[0]-x[1]*x[1])]), 
			lambda x: 1/2*(x[2]**2 + x[3]**2) +1/2*(x[0]**2+x[1]**2)+1*(x[0]**2 *x[1] -(x[1]**3)/3), 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,2], x[:,3]]), 
			lambda x: np.stack([-x[:,0]-2*1*x[:,0]*x[:,1], -x[:,1]-1*(x[:,0]*x[:,0]-x[:,1]*x[:,1])]), 
			lambda x: 1/2*(x[:,2]**2 + x[:,3]**2) +1/2*(x[:,0]**2+x[:,1]**2)+1*(x[:,0]**2 *x[:,1] -(x[:,1]**3)/3),
			lambda x: 1/2*(x[:,0]**2+x[:,1]**2)+1*(x[:,0]**2 *x[:,1] -(x[:,1]**3)/3),
			lambda x: 1/2*(x[:,2]**2 + x[:,3]**2),
			4, 32, 1, 2, 22, 22, 1, 0., 0., 0.01, 0.01, [np.pi,0.])

todalattice = System(lambda x: np.asarray([x[3], x[4], x[5]]), 
			lambda x: np.asarray([-np.exp(x[0]-x[1])+np.exp(x[2]-x[0]),
                           -np.exp(x[1]-x[2])+np.exp(x[0]-x[1]),
                           -np.exp(x[2]-x[0])+np.exp(x[1]-x[2]),]), 
			lambda x: 0.5*(x[3]**2+x[4]**2+x[5]**2)+np.exp(x[0]-x[1])+np.exp(x[1]-x[2])+np.exp(x[2]-x[0])-3, 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,3], x[:,4], x[:,5]]), 
			lambda x: np.stack([-np.exp(x[:,0]-x[:,1])+np.exp(x[:,2]-x[:,0]),
                           -np.exp(x[:,1]-x[:,2])+np.exp(x[:,0]-x[:,1]),
                           -np.exp(x[:,2]-x[:,0])+np.exp(x[:,1]-x[:,2]),]), 
			lambda x: 0.5*(x[:,3]**2+x[:,4]**2+x[:,5]**2)+np.exp(x[:,0]-x[:,1])+np.exp(x[:,1]-x[:,2])+np.exp(x[:,2]-x[:,0])-3,
			lambda x: np.exp(x[:,0]-x[:,1])+np.exp(x[:,1]-x[:,2])+np.exp(x[:,2]-x[:,0])-3,
			lambda x: 0.5*(x[:,3]**2+x[:,4]**2+x[:,5]**2),
			6, 31, 1, 3, 22, 22, 1, 0., 0., 0.01, 0.01, [np.pi,0.])

coupledoscillator = System(lambda x: np.asarray([x[3], x[4], x[5]]), 
			lambda x: np.asarray([1*(x[1]-x[0]), 1*(x[0]+x[2]) - 2*1*x[1], -1*(x[2]-x[1])]), 
			lambda x: 0.5*(x[3]**2 + x[4]**2 + x[5]**2) + 0.5* 1 *(x[1]-x[0])**2 + 0.5* 1 *(x[2]-x[1])**2 , 
			[(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)], 
			lambda x: np.stack([x[:,3],x[:,4],x[:,5]]), 
			lambda x: np.stack([1*(x[:,1]-x[:,0]), 1*(x[:,0]+x[:,2]) - 2*1*x[:,1], -1*(x[:,2]-x[:,1])]), 
			lambda x: 0.5*(x[:,3]**2 + x[:,4]**2 + x[:,5]**2) + 0.5* 1 *(x[:,1]-x[:,0])**2 + 0.5* 1 *(x[:,2]-x[:,1])**2,
			lambda x: 0.5* 1 *(x[:,1]-x[:,0])**2 + 0.5* 1 *(x[:,2]-x[:,1])**2,
			lambda x: 0.5*(x[:,3]**2 + x[:,4]**2 + x[:,5]**2), 
			6, 31, 1, 3, 22, 22, 1, 0., 0., 0.01, 0.01, [np.pi,0.])

print("imported systems")
