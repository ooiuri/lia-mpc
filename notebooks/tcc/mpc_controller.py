import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.interpolate import interp1d
from scipy.spatial.distance import directed_hausdorff

import matplotlib as mpl
import matplotlib.pyplot as plt


# import scienceplots
# plt.style.use('science')

class DifferentialDriveRobot:
    def __init__(self, R, L):
        self.R = R  # Raio da roda
        self.L = L  # Distância entre as rodas
        
        # Estados iniciais
        self.x = 0
        self.y = 0
        self.theta = 0

    def update(self, v_R, v_L, dt):
        v = (self.R / 2) * (v_R + v_L)
        omega = (self.R / self.L) * (v_R - v_L)
        
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt


class MPCController:
    """
    ### MPC Controller Class \n
    :param DifferentialDriveRobot robot: A robot model
    :param int N: Prediction Horizon
    :param float Q: Penalty for tracking error
    :param float R: Penalty for control effort
    :param float dt: Time interval
    :param max_margin: A margin of robot maximum offset
    """
    def __init__(self, robot, N, Q, R, dt, max_margin):
        self.robot = robot
        self.N = N  
        self.Q = Q  
        self.R = R  
        self.dt = dt
        self.max_margin = max_margin  # Distância máxima permitida da trajetória de referência

    def solve(self, x_ref, y_ref):
        model = pyo.ConcreteModel()

        model.N = self.N
        model.dt = self.dt
        model.Q = self.Q
        model.R = self.R
        model.x_ref = x_ref
        model.y_ref = y_ref

        # Variáveis de otimização
        model.v_R = pyo.Var(range(model.N), domain=pyo.Reals)
        model.v_L = pyo.Var(range(model.N), domain=pyo.Reals)

        # Estados
        model.x = pyo.Var(range(model.N+1), domain=pyo.Reals, initialize=0)
        model.y = pyo.Var(range(model.N+1), domain=pyo.Reals, initialize=0)
        model.theta = pyo.Var(range(model.N+1), domain=pyo.Reals, initialize=0)

        # Estados iniciais
        model.x[0].fix(self.robot.x)
        model.y[0].fix(self.robot.y)
        model.theta[0].fix(self.robot.theta)

        # Variável auxiliar para a Distância de Hausdorff
        model.hausdorff_dist = pyo.Var()

        # Função custo
        def objective_rule(model):
            cost = 0
            for t in range(model.N):
                cost += model.Q * ((model.x[t] - model.x_ref[t])**2 + (model.y[t] - model.y_ref[t])**2)
                # cost += model.R * (model.v_R[t]**2 + model.v_L[t]**2)
                cost += model.R * ((model.v_R[t] - model.v_L[t])**2)
            return cost
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Restrições de dinâmica
        model.dynamics = pyo.ConstraintList()
        for t in range(model.N):
            v = (self.robot.R / 2) * (model.v_R[t] + model.v_L[t])
            omega = (self.robot.R / self.robot.L) * (model.v_R[t] - model.v_L[t])
            model.dynamics.add(model.x[t+1] == model.x[t] + v * pyo.cos(model.theta[t]) * model.dt)
            model.dynamics.add(model.y[t+1] == model.y[t] + v * pyo.sin(model.theta[t]) * model.dt)
            model.dynamics.add(model.theta[t+1] == model.theta[t] + omega * model.dt)

        # Limites das variáveis de controle
        def control_limits_rule_R(model, t):
            return (-1, model.v_R[t], 1)
        
        def control_limits_rule_L(model, t):
            return (-1, model.v_L[t], 1)

        model.control_limits_R = pyo.Constraint(range(model.N), rule=control_limits_rule_R)
        model.control_limits_L = pyo.Constraint(range(model.N), rule=control_limits_rule_L)

        # Hausdorff_constraint of margin
        def hausdorff_constraint(model):
            # Listas de pontos da trajetória atual e da referência
            traj_pred = [(model.x[i].value, model.y[i].value) for i in range(self.N)]
            traj_ref = [(model.x_ref[i], model.y_ref[i]) for i in range(self.N)]

            # Calcula a Distância de Hausdorff em ambas direções
            hd_dist_1 = directed_hausdorff(traj_pred, traj_ref)[0]
            hd_dist_2 = directed_hausdorff(traj_ref, traj_pred)[0]
            
            # Define a Distância de Hausdorff como o máximo entre as duas direções
            model.hausdorff_dist.set_value(max(hd_dist_1, hd_dist_2))

            # Restrições para garantir que a Distância de Hausdorff fique dentro do limite
            return model.hausdorff_dist <= self.max_margin

        # Adiciona a restrição de Hausdorff ao modelo
        model.hausdorff_constraint = pyo.Constraint(rule=hausdorff_constraint)

        # Solver
        solver = SolverFactory('ipopt')
        result = solver.solve(model, tee=False)

        # Extrair os valores de controle resultantes
        out_X = [model.x[i]() for i in range(self.N)]
        out_Y = [model.y[i]() for i in range(self.N)]

        out_v_R = [model.v_R[i]() for i in range(self.N)]
        out_v_L = [model.v_L[i]() for i in range(self.N)]
        print('out_v_R', out_v_R)
        print('out_v_L', out_v_L)
        return model.v_R[0].value, model.v_L[0].value, [out_X, out_Y]


# Parâmetros do robô
R = 0.1  # Raio da roda em metros
L = 0.5  # Distância entre as rodas em metros

# Criar instância do robô diferencial
robot = DifferentialDriveRobot(R, L)

# Criar instância do controlador MPC
predict_horizon = 10
margin = 0.1  # Distância da margem em metros

mpc = MPCController(robot, N=predict_horizon, Q=1, R=0, dt=8, max_margin=margin)

# Referência de trajetória
in_points = [[-0.09150369689456136, 0.48914467381475923], [-0.09846197449972373, 0.5594131906773301], [-0.10708867633569592, 0.6468840483426161], [-0.11807990084029595, 0.7587300914046574], [-0.13122400333321313, 0.9066888477317598], [-0.15099370681792523, 1.111769544992076]]

Xc = []
Yc = []
for point in in_points:
    Xc.append(point[1])
    Yc.append(point[0])

print('Xc: ', Xc)
print('Yc: ', Yc)
style_path = "paper.mplstyle"
print(f"Beginning of file {style_path}:")
with open(style_path) as f:
    lines_to_print = 8
    for i in range(lines_to_print):
        line = f.readline()
        print(line, end='')

print("...")

plt.style.use("./paper.mplstyle")

plt.figure(figsize=(8, 6))
plt.plot(Xc, Yc, 'o-', label='Trajetória Detectada', markersize=8)


# Adicionando linhas verticais e horizontais passando pela origem (0, 0)
plt.axhline(0, color='gray', linewidth=1, linestyle='--')  # Linha horizontal
plt.axvline(0, color='gray', linewidth=1, linestyle='--')  # Linha vertical
plt.xlabel('Distância em x [m]')
plt.ylabel('Distância em y [m]')
plt.title('Plotagem dos pontos detectados')
plt.legend()
plt.grid(True)

plt.savefig(f"aaa.pdf")
plt.show()

# Cria uma trajetória suave do ponto (0, 0) até o primeiro ponto (Xc[0], Yc[0])
# x_initial = np.linspace(0, Xc[0], 5)
# y_initial = np.linspace(0, Yc[0], 5)

# # x_initial = [0]
# # y_initial = [0]
# # Concatena a trajetória inicial com os pontos de referênci
# x_initial = np.concatenate((Xc, 10 * [Xc[-1]]))
# y_initial = np.concatenate((Yc, 10 * [Yc[-1]]))
# print('x_initial: ', x_initial)
# # x_ref = np.concatenate((x_ref, 20*[Xc[-1]]))
# # y_ref = np.concatenate((y_ref, 20*[Yc[-1]]))

# # Cria uma amostragem mais densa da trajetória usando interpolação
# num_intermediate_points = 2  # Quantidade de pontos entre cada ponto de Xc e Yc
# t_original = np.linspace(0, 1, len(x_initial))  # Tempo para os pontos originais
# t_dense = np.linspace(0, 1, len(x_initial) * num_intermediate_points)  # Tempo mais denso para interpolação

# # Interpoladores para x_initial e Yc
# x_interp_func = interp1d(t_original, x_initial, kind='cubic')
# y_interp_func = interp1d(t_original, y_initial, kind='cubic')

# # Trajetória interpolada
# x_ref = x_interp_func(t_dense)
# y_ref = y_interp_func(t_dense)

# # # Simulação
# dt = 8
# x_traj = [0]
# y_traj = [0]
# v_out = []

# theta_ref = np.arctan2(np.gradient(y_ref), np.gradient(x_ref))
# x_margin_sup = x_ref + margin * np.cos(theta_ref + np.pi/2)
# y_margin_sup = y_ref + margin * np.sin(theta_ref + np.pi/2)
# x_margin_inf = x_ref - margin * np.cos(theta_ref + np.pi/2)
# y_margin_inf = y_ref - margin * np.sin(theta_ref + np.pi/2)

# print('iniciando simulação')
# print('len xref', len(x_ref))


# for i in range(len(x_ref) - predict_horizon):  # Garantir que o loop seja executado
#     v_R, v_L, [out_x, out_y] = mpc.solve(x_ref[i:i+predict_horizon], y_ref[i:i+predict_horizon])
#     v_out.append((v_R, v_L))
#     robot.update(v_R, v_L, dt)
#     x_traj.append(robot.x)
#     y_traj.append(robot.y)

#     # Plotar trajetória com margem
#     plt.plot(x_ref, y_ref, 'o', label='Trajetória de Referência')
#     plt.plot(out_x, out_y, label='Trajetória Prevista')
#     plt.plot(x_traj, y_traj, label='Trajetória do Robô')
#     plt.fill_between(x_ref, y_margin_inf, y_margin_sup, color='gray', alpha=0.3, label=f'Margem de {margin} m')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend()
#     # plt.show()
#     plt.savefig(f'./assets/test4/image_{i}')
#     plt.clf() 

# # v_R, v_L, out_x, out_y = mpc.solve(x_ref[0:predict_horizon], y_ref[0:predict_horizon])
# # v_out.append((v_R, v_L))
# # robot.update(v_R, v_L, dt)
# # x_traj.append(robot.x)
# # y_traj.append(robot.y)

# print('v_out', v_out)
# print('v_out.len', len(v_out))





