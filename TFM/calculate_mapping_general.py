#from IPython import get_ipython
#get_ipython().magic('reset -sf')

import numpy as np
import numpy.matlib
from scipy.optimize import linprog  
# CVX
import cvxpy as cvx

# MAPPING
#   
# P = MAPPING(algorithm, X, V, W, vector_norm, chosen_variable) 
# calcula los mapeos de varios métodos de ejes radiales, según el algoritmo, se define el método radial
# X es una matriz de N por n cuyas filas contienen las n muestras de datos dimensionales.
# V es una matriz de n por m cuyas filas definen los vectores de eje del método. 
# W es una matriz n por n diagonal que define ponderaciones no negativas para cada variable.
# vector_norm es la norma vectorial asociada a los diagramas de ejes radiales adaptables
# chosen_variable es el atributo seleccionado para los gráficos de ejes radiales adaptables restringidos.
# Las incrustaciones de baja dimensión se almacenan en la N mediante la matriz P.

def mapping(algorithm,X,V,W,vector_norm,chosen_variable):
    X = np.matrix(X)
    V = np.matrix(V)
    W = np.matrix(W)
	
    [N,n] = X.shape
    m = V.shape[1]
    if algorithm == 'SC':
        P = X * V
    elif algorithm == 'RadViz':
        minimum = np.amin(X, axis=0)
        maximum = np.amax(X ,axis=0)        
        X_Radviz = (X - numpy.matlib.repmat(minimum, N, 1)) / (numpy.matlib.repmat(maximum, N, 1) - numpy.matlib.repmat(minimum, N, 1))     
        for i in range(N):
            sum_row = np.sum(X_Radviz[i,])
            if sum_row == 0:
                X_Radviz[i,] = np.ones((n)) / n
            else:
                X_Radviz[i,] = X_Radviz[i,] / sum_row        
        P = X_Radviz * V    
    
    elif algorithm == 'SRA':
        for i in range(n):
            norm_squared_of_ith_row_of_V = V[i,] * V[i,].transpose()
            if norm_squared_of_ith_row_of_V.any() != 0:
                V[i,] = V[i,] / norm_squared_of_ith_row_of_V 
        P = X * (np.linalg.pinv(W * V) * W).transpose()
        
    elif algorithm == 'Adaptable':
        if vector_norm == 1:
            A = np.vstack((np.hstack((-np.identity(n) , -W * V)), np.hstack((-np.identity(n) , W * V))))  
            f = np.hstack((np.ones((n)) , np.zeros((m))))
            P = np.zeros((N,m))
            for i in range(N):
                b = np.vstack((-W * X[i,].transpose(), W * X[i,].transpose()))
                p_star = linprog(f, A_ub=A,b_ub=b,bounds=(float('-inf'),None))     
                P[i,] = p_star.x[-m:].transpose()
        
        elif vector_norm == 'Inf':
            A = np.vstack((np.hstack((-np.ones((n,1)), -W * V)), np.hstack((-np.ones((n,1)), W * V))))	
            f = np.hstack((1, np.zeros((m))))
            P = np.zeros((N, m))  
            for i in range(N):    
                b = np.vstack((-W * X[i,].transpose() , W * X[i,].transpose()))
                p_star = linprog(f, A_ub=A, b_ub=b, bounds=(float('-inf'), None))     
                P[i,] = p_star.x[-m:].transpose()
                
        else:
            #for i in range(n):
            #norm_squared_of_ith_row_of_V = V[i,] * V[i,].transpose()
            #if norm_squared_of_ith_row_of_V.any() != 0:
            #    V[i,] = V[i,] / norm_squared_of_ith_row_of_V 
            P = X * (np.linalg.pinv(W * V) * W).transpose()
        
        
					
    elif algorithm == 'Adaptable exact':
        v = np.squeeze(np.asarray(V[chosen_variable,]))
        x = np.squeeze(np.asarray(X[:,chosen_variable]))
        if vector_norm == 1:
            A = np.vstack((np.hstack((-np.identity(n), -V)),np.hstack((-np.identity(n), V)))) 
            f = np.hstack((np.ones((n)), 0, 0))
            Aeq = np.matrix([np.hstack((np.zeros((n)), v))])
            P = np.zeros((N, 2))
            for i in range(N):
                b = np.vstack((-X[i,].transpose(),X[i,].transpose()))
                p_star = linprog(f, A, b, Aeq, np.array([x[i]]), bounds=(float('-inf'), None)) 
                if p_star:
                    P[i,] = p_star.x[-m:].transpose()
                    
        elif vector_norm == 2:
            P = cvx.Variable(N, 2)
            obj = cvx.Minimize(cvx.norm(P * V.T - X, "fro"))
            constraints = [P * v.T == x]
            prob = cvx.Problem(obj, constraints)
            prob.solve()
            P = P.value
			
        elif vector_norm == 'Inf':
            A = np.vstack((np.hstack((-np.ones((n, 1)), -V)), np.hstack((-np.ones((n, 1)), V))))	
            f = np.hstack((1, np.zeros((2))))
            Aeq = np.matrix([np.hstack((0, v))])
            P = np.zeros((N, 2))
            for i in range (N):
                b = np.vstack((-X[i,].transpose(), X[i,].transpose()))	
                p_star = linprog(f, A, b, Aeq, x[i], bounds=(float('-inf'), None))
                if p_star:
                    P[i,] = p_star.x[-m:].transpose()
            			
    elif algorithm == 'Adaptable ordered':
        k = chosen_variable
        I = np.argsort(np.asarray(X)[:,k])
        constraints_list = []
        if vector_norm == 1:
            P = cvx.Variable(N, 2)
            z = 0
            for i in range(N):
                z = z + cvx.norm(V * P[i,:].T - X[i,].T, 1)
            obj = cvx.Minimize(z)
            for i in range (N-1):
                constraints_list.append(P[I[i],:] * V[k,].T <= P[I[i+1],:] * V[k,].T)
            prob = cvx.Problem(obj, constraints_list)
            prob.solve()
            P = P.value
	
        elif vector_norm == 2:
            P = cvx.Variable(N, 2)
            obj = cvx.Minimize(cvx.norm(P * V.T - X, "fro"))
            for i in range (N-1):
                constraints_list.append(P[I[i],:] * V[k,].T <= P[I[i+1],:] * V[k,].T)	
            prob = cvx.Problem(obj, constraints_list)    
            prob.solve()
            P = P.value
			
        elif vector_norm == 'Inf':
            P = cvx.Variable(N, 2)
            x = cvx.Variable()
            obj = cvx.Minimize(x)
            for j in range(n):
                z = cvx.norm(P * V[j,].T - X[:,j], "inf")
                constraints_list.append(z <= x)                
            for i in range(N-1):
                constraints_list.append(P[I[i],:] * V[k,].T <= P[I[i+1],:] * V[k,].T)
            prob = cvx.Problem(obj, constraints_list)
            prob.solve()
            P = P.value   
            
    return(P)     


#X = np.loadtxt(r"C:\Users\lramoari\Desktop\TFG\cereal_data_no_missing.txt")
#V = np.loadtxt(r'C:\Users\lramoari\Desktop\TFG\V_Var.txt')
#W = np.identity(11)

#k = 0
#P = mapping('Adaptable exact', X, V, W, 'Inf', k)
#print(P)

#    #np.random.seed(1)
#    #N = 100
#    #n = 5
#    #X = np.random.randn(N,n)
#    #V = np.random.randn(n,2)
#    #k = math.ceil(np.random.randn(1)*n)
#    #W = np.diag(np.random.randn(n,1)*n)
