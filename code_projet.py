# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:56:01 2026

@author: james
"""


import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time
import warnings

# =========================================================
# 1. PARAMÈTRES GLOBAUX INITIAUX
# =========================================================

# Végétation (V)
r_V = 1.20       # Taux de croissance intrinsèque
K_V = 500.0      # Capacité de charge
alpha_N = 15.0   # Intensité de pâturage wapitis/orignaux
nu_N = 200.0     # Demi-saturation pâturage wapitis/orignaux
alpha_D = 8.0    # Intensité de pâturage cerfs
nu_D = 100.0     # Demi-saturation pâturage cerfs
alpha_B = 3.0    # Intensité de pâturage ours
nu_B = 120.0     # Demi-saturation pâturage ours

# Ongulés principaux : Wapitis / Orignaux (N)
r_N = 0.30       # Taux de croissance intrinsèque
K_N = 7.0        # Capacité de charge maximale
V_star = 70.0    # Demi-saturation pour kappa(V)
theta_N = 4.0    # Exposant theta-logistique
c_WN = 7.5       # Coef. mise à mort loup-wapiti
beta = 0.5       # Paramètre de dépendance au ratio
c_BN = 0.5       # Coef. mise à mort ours-wapiti
h_BN = 3.0       # Demi-saturation ours-wapiti
delta_N = 0.05   # Mortalité de fond

# Ongulés secondaires : Cerfs (D)
r_D = 0.50       # Taux de croissance intrinsèque
K_D = 4.0        # Capacité de charge maximale
V_star_star = 50.0 # Demi-saturation pour kappa_D(V) (noté V** dans le sujet)
theta_D = 2.0    # Exposant theta-logistique
c_WD = 2.0       # Coef. mise à mort loups-cerfs
h_WD = 2.0       # Demi-saturation loups-cerfs
c_BD = 0.3       # Coef. mise à mort ours-cerfs
h_BD = 1.5       # Demi-saturation ours-cerfs
delta_D = 0.08   # Mortalité de fond

# Loups (W)
epsilon_1 = 0.25 # Pente logarithmique réponse numérique
epsilon_2 = 0.5  # Décalage de base réponse numérique
eta = 0.01       # Décalage de régularisation log
m_W = 0.15       # Mortalité naturelle
mu_W = 0.10      # Amplitude prélèvement humain saisonnier

# Ours (B)
e_B = 0.12       # Gain énergétique depuis proies
e_V = 0.02       # Gain énergétique depuis végétation
m_B = 0.08       # Mortalité naturelle

# =========================================================
# 2. CONDITIONS INITIALES DE BASE (t=0)
# =========================================================

V0 = 440.0
N0 = 4.5
D0 = 2.5
W0 = 0.04
B0 = 0.25
u0 = np.array([V0, N0, D0, W0, B0])

# =========================================================
# 3. DÉFINITION DES MÉTHODES ET SOLVEURS (Version 1)
# =========================================================

def F(t, u):
    
    V, N, D, W, B = u
   
    
    kappa_N = K_N * (V / (V_star + V))
    kappa_D = K_D * (V / (V_star_star + V))
    phi_W = ((c_WN * N) / (W + beta * N)) + ((c_WD * D) / (h_WD + D))
   
    dVdt = r_V * V * (1 - V / K_V) - ((alpha_N * V * N) / (nu_N + V)) - ((alpha_D * V * D) / (nu_D + V)) - ((alpha_B * V * B) / (nu_B + V))
    dNdt = r_N * N * (1 - (N / kappa_N)**theta_N) - ((c_WN * N * W) / (W + beta * N)) - ((c_BN * N * B) / (h_BN + N)) - delta_N * N
    dDdt = r_D * D * (1 - (D / kappa_D)**theta_D) - ((c_WD * D * W) / (h_WD + D)) - ((c_BD * D * B) / (h_BD + D)) - delta_D * D
    dWdt = (epsilon_1 * np.log(phi_W + eta) - epsilon_2) * W - m_W * W - mu_W * np.sin(np.pi * t)**2 * W
    dBdt = (e_B * (((c_BN * N) / (h_BN + N)) + ((c_BD * D) / (h_BD + D))) + (e_V * alpha_B * V) / (nu_B + V) - m_B) * B
   
    return np.array([dVdt, dNdt, dDdt, dWdt, dBdt])

def resolv_systeme(t, dt, U0, Fn , Maxiter= 1000) :
    
    interval_de_temps=[0]
    
    k=0
    U_solution = [U0]
   
    while(k < Maxiter and interval_de_temps[-1]<t) :
        Tsuivant = interval_de_temps[k] + dt
        
        interval_de_temps.append(Tsuivant)
        def G(X) :
            return X-U_solution[k]-dt*Fn(interval_de_temps[k+1],X)
       
        X= fsolve(G,U_solution[k])      
        
        U_solution.append(X.copy())
        k=k+1
       
    return interval_de_temps , U_solution, k

def decomposition_PA_LU(A):
    n = len(A)
    
    U = np.array(A, dtype=float)
    L = np.eye(n)
    P = np.eye(n)
   
    for k in range(n - 1):
        pivot_idx = np.argmax(np.abs(U[k:, k])) + k
       
        if pivot_idx != k:
            U[[k, pivot_idx], :] = U[[pivot_idx, k], :]
            
            P[[k, pivot_idx], :] = P[[pivot_idx, k], :]
           
            if k > 0:
                L[[k, pivot_idx], :k] = L[[pivot_idx, k], :k]
               
        for i in range(k + 1, n):
            
            if U[k, k] == 0:
                
                raise ZeroDivisionError("Pivot nul détecté, la matrice est singulière.")
           
            alpha = U[i, k] / U[k, k]
            L[i, k] = alpha
            U[i, k:] = U[i, k:] - alpha * U[k, k:]
           
    return P, L, U

def descente(L, b):
    
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y

def remontee(U, y):
    n = len(U)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x

def resoudre_PA_LU(A, b):
    P, L, U = decomposition_PA_LU(A)
    b_permute = np.dot(P, b)
    y = descente(L, b_permute)
    x = remontee(U, y)
    
    return x

def Jacobien(G,x,eps=10**(-6)) :
    
    n=len(x)
    J=np.zeros((n,n))
    G0=G(x)
    for j in range(n):
        
       x_var=x.copy()
       x_var[j]=x_var[j]+eps
       G_var=G(x_var)
       J[:,j]=(G_var-G0)/eps
    return J    

def solveurMaison (G,Un,Tsuivant,dt,maxiter=50,tol=1e-6 ) :
    
    Uk=Un.copy()
    for k in range(maxiter):
        
        Gk=G(Uk)
        if(np.linalg.norm(Gk)<= tol):
            
            return Uk
        J=Jacobien(G,Uk)
        invJ= resoudre_PA_LU(J,Gk)
        Uk=Uk - invJ
    return Uk

def resolv_systeme_maison(t, dt, U0, Fn , Maxiter= 1000) :
    interval_de_temps=[0]
    k=0
    U_solution = [U0]
   
    while(k < Maxiter and interval_de_temps[-1]<t) :
        
        Tsuivant = interval_de_temps[k] + dt
        interval_de_temps.append(Tsuivant)
        def G(X) :
            
            return X-U_solution[k]-dt*Fn(interval_de_temps[k+1],X)
       
        X= solveurMaison(G,U_solution[k],interval_de_temps[k+1],dt)      
        U_solution.append(X.copy())
        k=k+1
       
    return interval_de_temps , U_solution, k




def eulerexplicite(t, dt, U0, Fn , Maxiter= 1000):
    
    n = U0.shape[0]
    interval_de_temps=[0]
    k=0
    U_solution = [U0]
    while(k < Maxiter and interval_de_temps[-1]<t) :
        
        X = U0.copy()
        Tsuivant = interval_de_temps[k] + dt
        interval_de_temps.append(Tsuivant)
        for i in range(n):
            X[i] = U_solution[k][i] + dt * Fn(interval_de_temps[k], U_solution[k])[i]
        U_solution.append(X.copy())
        k +=1
    return interval_de_temps , U_solution, k



def rungekutta5(t, dt, U0, Fn ,Maxiter= 1000):
    interval_de_temps=[0]
    k=0
    U_solution = [U0]
    while(k < Maxiter and interval_de_temps[-1]<t) :
        
        Tsuivant = interval_de_temps[k] + dt
        interval_de_temps.append(Tsuivant)
        k1 = Fn(interval_de_temps[k], U_solution[k])
        k2 = Fn(interval_de_temps[k] + 1/4 * dt, U_solution[k] + dt*1/4*k1)
        k3 = Fn(interval_de_temps[k] + 1/4 * dt, U_solution[k] + dt*1/8*(k1 + k2))
        k4 = Fn(interval_de_temps[k] + 1/2 * dt, U_solution[k] + dt*(-1/2 * k2 + k3))
        k5 = Fn(interval_de_temps[k] + 3/4 * dt, U_solution[k] + dt*(3/16*k1 + 9/16 * k4))
        k6 = Fn(interval_de_temps[k] + dt, U_solution[k] + dt * (-3/7 *k1 + 2/7 * k2 + 12/7 * k3 - 12/7 * k4 + 8/7 * k5))
        X = U_solution[k] + dt * (7/90 * k1 + 32/90 * k3 + 12/90 * k4 + 32/90 * k5 + 7/90 * k6)
        U_solution.append(X.copy())
        k += 1
    return interval_de_temps , U_solution, k  


#%% IMPLICITE NUMPY
U0 = np.array([440.0, 35.0, 2.5, 0.04, 0.25])
start_time1 = time.time()

inerval_temp, solution, iteration_scipy = resolv_systeme(50, 0.01, U0, F, Maxiter= 10000)
end_time1 = time.time()

time_eleur_implicite_numpy = end_time1 - start_time1
solution = np.array(solution)

plt.figure()
labels = ["Végétation", "Wapitis/Orignaux", "Cerfs", "Loups", "Ours"]
for i in range(5):
    plt.plot(inerval_temp, solution[:,i],label=labels[i])
   
plt.title("Dynamique des populations au cours du temps avec solveur scipy")
plt.xlabel("Temps")
plt.ylabel("Populations")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

#%% IMPLICITE MAISON
U0 = np.array([440.0, 35.0, 2.5, 0.04, 0.25])
start_time2 = time.time()

inerval_temp, solution, iteration_implicite = resolv_systeme_maison(50, 0.01, U0, F, Maxiter= 10000)
end_time2 = time.time()

time_eleur_implicite_maison = end_time2 - start_time2
solution = np.array(solution)

plt.figure()
for i in range(5):
    plt.plot(inerval_temp, solution[:,i],label=labels[i])
   
plt.title("IMPLICITE")
plt.xlabel("Temps")
plt.ylabel("Populations")
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.show()

#%% EXPLICITE
U0 = np.array([440.0, 35.0, 2.5, 0.04, 0.25])
start_time3 = time.time()

inerval_temp, solution, iteration_explicite = eulerexplicite(50, 0.01, U0, F, Maxiter= 10000)
end_time3 = time.time()

time_eleur_explicite = end_time3 - start_time3
solution = np.array(solution)



#%% RUNGE KUTTA
U0 = np.array([440.0, 35.0, 2.5, 0.04, 0.25])
start_time4 = time.time()

inerval_temp, solution, iteration_rungekutta = rungekutta5(50, 0.01, U0, F, Maxiter= 10000)
end_time4 = time.time()

time_rugkkuta = end_time4 - start_time4
solution = np.array(solution)



#%% TEST TEMPS ET ITÉRATION
algo = ['scipy methode', 'euler implicite', 'euleur explicite', 'rungekutta']
algo_temps = [time_eleur_implicite_numpy, time_eleur_implicite_maison, time_eleur_explicite, time_rugkkuta]
algo_iteration = [iteration_scipy, iteration_implicite, iteration_explicite, iteration_rungekutta]
   
plt.figure()
plt.bar(algo, algo_temps)
plt.title("temps de calcul des algorithmes")


#%% COMPARAISON DES STABILITÉS 



warnings.filterwarnings("ignore", category=RuntimeWarning) 

T_test = 2.0          
dt_difficile = 0.02  
U0_choc = np.array([440.0, 23.0, 2.5, 0.04, 0.25]) 

print(f"Lancement du test de stabilité (dt = {dt_difficile})...")

t_imp, sol_imp, _ = resolv_systeme_maison(T_test, dt_difficile, U0_choc, F, Maxiter=10000)
sol_imp = np.array(sol_imp)

t_rk, sol_rk, _ = rungekutta5(T_test, dt_difficile, U0_choc, F, Maxiter=10000)
sol_rk = np.array(sol_rk)

t_exp, sol_exp, _ = eulerexplicite(T_test, dt_difficile, U0_choc, F, Maxiter=10000)
sol_exp = np.array(sol_exp)

plt.figure(figsize=(10, 6))
indice_pop = 1 
nom_pop = "Wapitis/Orignaux"

plt.plot(t_imp, sol_imp[:, indice_pop], label="Euler Implicite (Stable)", 
         color="tab:green", linewidth=2.5)

plt.plot(t_rk, sol_rk[:, indice_pop], label="Runge-Kutta 5", 
         color="tab:orange", linestyle="--", linewidth=2, marker='o', markersize=4)

plt.plot(t_exp, sol_exp[:, indice_pop], label="Euler Explicite (Divergence rapide)", 
         color="tab:red", linestyle=":", linewidth=1.5, marker='x', markersize=5)

plt.ylim(-20, 100) 
plt.xlim(0, T_test)

plt.title(f"Démonstration de l'instabilité numérique ($N_0={U0_choc[1]}$, $dt={dt_difficile}$)")
plt.xlabel("Temps (années)")
plt.ylabel(f"Population de {nom_pop}")
plt.legend(loc="upper right")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.axhline(0, color='black', linewidth=1, alpha=0.8)

plt.savefig("comparaison_stabilites_lisible_2.png", dpi=300, bbox_inches='tight')

T_test = 2.0          
dt_difficile = 0.02  
U0_choc = np.array([440.0, 35.0, 2.5, 0.04, 0.25]) 

print(f"Lancement du test de stabilité (dt = {dt_difficile})...")

t_imp, sol_imp, _ = resolv_systeme_maison(T_test, dt_difficile, U0_choc, F, Maxiter=10000)
sol_imp = np.array(sol_imp)

t_rk, sol_rk, _ = rungekutta5(T_test, dt_difficile, U0_choc, F, Maxiter=10000)
sol_rk = np.array(sol_rk)

t_exp, sol_exp, _ = eulerexplicite(T_test, dt_difficile, U0_choc, F, Maxiter=10000)
sol_exp = np.array(sol_exp)

plt.figure(figsize=(10, 6))
indice_pop = 1 
nom_pop = "Wapitis/Orignaux"

plt.plot(t_imp, sol_imp[:, indice_pop], label="Euler Implicite (Stable)", 
         color="tab:green", linewidth=2.5)

plt.plot(t_rk, sol_rk[:, indice_pop], label="Runge-Kutta 5", 
         color="tab:orange", linestyle="--", linewidth=2, marker='o', markersize=4)

plt.plot(t_exp, sol_exp[:, indice_pop], label="Euler Explicite (Divergence rapide)", 
         color="tab:red", linestyle=":", linewidth=1.5, marker='x', markersize=5)


plt.ylim(-20, 100) 
plt.xlim(0, T_test)

plt.title(f"Démonstration de l'instabilité numérique ($N_0={U0_choc[1]}$, $dt={dt_difficile}$)")
plt.xlabel("Temps (années)")
plt.ylabel(f"Population de {nom_pop}")
plt.legend(loc="upper right")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.axhline(0, color='black', linewidth=1, alpha=0.8)



plt.show()


#%% CAS DE BASE (200 ANS)
T = 200 #Nombre d'année
dT = 0.05 #Pas de temps
U0 = np.array([440, 4.5, 2.5, 0.04, 0.25])
interval_temp, solution, _ = resolv_systeme_maison(T, dT, U0, F, Maxiter = 10000)
solution = np.array(solution)

fig, ax1 = plt.subplots(figsize=(10, 6))

couleur_veg = 'tab:green'
ax1.set_xlabel("Temps (années)")
ax1.set_ylabel("Végétation (kg/km²)", color=couleur_veg)
ax1.plot(interval_temp, solution[:, 0], label="Végétation", color=couleur_veg, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=couleur_veg)
ax1.set_ylim(bottom=0, top=500)

ax2 = ax1.twinx()
ax2.set_ylabel("Densité des animaux (Échelle Linéaire Ajustée)")
couleurs_animaux = ['tab:orange', 'tab:blue', 'tab:red', 'tab:purple']

multiplicateurs = [1, 2, 100, 10]
noms_animaux = ["Wapitis/Orignaux", "Cerfs", "Loups", "Ours"]

for i in range(1, 5):
    mult = multiplicateurs[i-1]
    nom = noms_animaux[i-1]
    label_multiplie = f"{nom} (x{mult})"
    donnees_ajustees = solution[:, i] * mult
    ax2.plot(interval_temp, donnees_ajustees, label=label_multiplie, color=couleurs_animaux[i-1], linewidth=2)

ax2.set_ylim(bottom=0, top=25)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1.40, 1))

plt.title("Cas de base")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.show()

#%% CASCADE TROPHIQUE : PARAMÈTRES DE BASE
T_total = 200
dT = 0.05
Annee_reintroduction = 15

U0_initial = np.array([400.0, 2.0, 2.5, 0.0, 0.1]) 
inerval_temp_1, solution_1, _ = resolv_systeme_maison(Annee_reintroduction, dT, U0_initial, F, Maxiter=10000)
solution_1 = np.array(solution_1)

U_reintroduction = solution_1[-1] + np.array([0.0, 0.0, 0.0, 0.08, 0.0])
inerval_temp_2, solution_2, _ = resolv_systeme_maison(T_total - Annee_reintroduction, dT, U_reintroduction, F, Maxiter=10000)
solution_2 = np.array(solution_2)

inerval_temp_2_decale = np.array(inerval_temp_2) + Annee_reintroduction
temps_total = np.concatenate((inerval_temp_1, inerval_temp_2_decale))
solution_totale = np.vstack((solution_1, solution_2))

fig, ax1 = plt.subplots(figsize=(10, 6))

couleur_veg = 'tab:green'
ax1.set_xlabel("Temps (années)")
ax1.set_ylabel("Végétation (kg/km²)", color=couleur_veg)
ax1.plot(temps_total, solution_totale[:, 0], label="Végétation", color=couleur_veg, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=couleur_veg)
ax1.set_ylim(bottom=0, top=500) 

ax2 = ax1.twinx()  
ax2.set_ylabel("Densité des animaux ")
couleurs_animaux = ['tab:orange', 'tab:blue', 'tab:red', 'tab:purple']

multiplicateurs = [1, 2, 10, 50] 
noms_animaux = ["Wapitis/Orignaux", "Cerfs", "Loups", "Ours"]

for i in range(1, 5):
    mult = multiplicateurs[i-1]
    nom = noms_animaux[i-1]
    label_multiplie = f"{nom} (x{mult})"
    donnees_ajustees = solution_totale[:, i] * mult
    ax2.plot(temps_total, donnees_ajustees, label=label_multiplie, color=couleurs_animaux[i-1], linewidth=2)

ax2.set_ylim(bottom=0, top=25) 
ax1.axvline(x=Annee_reintroduction, color='red', linestyle='--', linewidth=2, label='Réintroduction (t=25)')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1.40, 1))

plt.title("Cascade Trophique : Paramètres de base")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.show()

#%% CASCADE TROPHIQUE : ÉQUILIBRE PARFAIT
# =========================================================
# PARAMÈTRES POUR FORCER LA BAISSE ET LA STABILISATION
# =========================================================

# 1. PARAMÈTRES DE LA VÉGÉTATION ET DES PROIES
r_V = 1.0        # Pousse de la végétation
alpha_N = 20.0   # Appétit des wapitis
r_N = 0.60       # Les wapitis se reproduisent assez vite pour survivre
K_N = 15.0       # Capacité d'accueil confortable pour les wapitis
K_D = 6.0        # Capacité des cerfs

# 2. LE REGLAGE FIN DES LOUPS (Le secret de l'équilibre)
c_WN = 2.0       # Efficacité de chasse modérée
c_WD = 0.5       # Chasse sur les cerfs réduite
beta = 0.8       # Interférence : les loups se gênent légèrement en chassant
epsilon_2 = 0.1  # Métabolisme ralenti : la meute survit avec moins de proies
m_W = 0.1        # Mortalité naturelle très faible
mu_W = 0.03      # Pas de chasse humaine (indispensable pour un équilibre parfait)
r_D = 0.6        # Croissance des cerfs, au lieu de 0.3

# 3. CONTRÔLE DES OURS
e_V = 0.02       
m_B = 0.11       # On augmente un peu leur mortalité pour qu'ils ne dévorent pas tout

# PARAMÈTRES DE LA SIMULATION
T_total = 200    
dT = 0.05
Annee_reintroduction = 25

# PHASE 1 : Avant les loups (0 à 20 ans) - Le surpâturage
U0_initial = np.array([400.0, 2.0, 2.5, 0.0, 0.1]) 
inerval_temp_1, solution_1, _ = resolv_systeme_maison(Annee_reintroduction, dT, U0_initial, F, Maxiter=10000)
solution_1 = np.array(solution_1)

# PHASE 2 : Réintroduction (20 à 200 ans) - Vers l'équilibre
U_reintroduction = solution_1[-1] + np.array([0.0, 0.0, 0.0, 0.08, 0.0])
inerval_temp_2, solution_2, _ = resolv_systeme_maison(T_total - Annee_reintroduction, dT, U_reintroduction, F, Maxiter=10000)
solution_2 = np.array(solution_2)

# Décalage du temps pour la phase 2 et fusion
inerval_temp_2_decale = np.array(inerval_temp_2) + Annee_reintroduction
temps_total = np.concatenate((inerval_temp_1, inerval_temp_2_decale))
solution_totale = np.vstack((solution_1, solution_2))

fig, ax1 = plt.subplots(figsize=(10, 6))

couleur_veg = 'tab:green'
ax1.set_xlabel("Temps (années)")
ax1.set_ylabel("Végétation (kg/km²)", color=couleur_veg)
ax1.plot(temps_total, solution_totale[:, 0], label="Végétation", color=couleur_veg, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=couleur_veg)
ax1.set_ylim(bottom=0, top=500) 

ax2 = ax1.twinx()  
ax2.set_ylabel("Densité des animaux")
couleurs_animaux = ['tab:orange', 'tab:blue', 'tab:red', 'tab:purple']

multiplicateurs = [1, 2, 10, 50] 
noms_animaux = ["Wapitis/Orignaux", "Cerfs", "Loups", "Ours"]

for i in range(1, 5):
    mult = multiplicateurs[i-1]
    nom = noms_animaux[i-1]
    label_multiplie = f"{nom} (x{mult})"
    donnees_ajustees = solution_totale[:, i] * mult
    ax2.plot(temps_total, donnees_ajustees, label=label_multiplie, color=couleurs_animaux[i-1], linewidth=2)

ax2.set_ylim(bottom=0, top=25) 
ax1.axvline(x=Annee_reintroduction, color='red', linestyle='--', linewidth=2, label='Réintroduction (t=25)')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', bbox_to_anchor=(1.40, 1))

plt.title("Cascade Trophique : Équilibre Parfait")
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.show()
