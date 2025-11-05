import torch
import torch.nn as nn

# Definição da Arquitetura da Rede Neural (PINN)
class FluxFieldPINN(nn.Module):
    def __init__(self, n_input=2, n_output=2, n_hidden=128, n_layers=6):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), nn.Tanh()])
        layers.append(nn.Linear(n_hidden, n_output))
        self.network = nn.Sequential(*layers)

    def forward(self, z, lambda_norm):
        # z: altura no dossel (normalizada ou não)
        # lambda_norm: comprimento de onda (normalizado entre -1 e 1)
        # O input deve ser um tensor de forma [N, 2]
        input_tensor = torch.cat([z, lambda_norm], dim=1)
        return self.network(input_tensor)

# Esboço da função de perda
def calculate_loss(network, LAI_learnable, R_solo_tensor, R_umida_tensor, wavelengths_tensor, H, k_func, s_func):
    """
    Calcula a perda total para o treinamento da PINN.
    Esta função precisa ser completada com as funções k_func e s_func.
    """
    w_pde, w_bc, w_data = 1.0, 1.0, 1.0

    # 1. Perda na Física (Resíduo da PDE)
    num_collocation_points = 10000
    z_collocation = H * torch.rand(num_collocation_points, 1, requires_grad=True)
    lambda_collocation = torch.rand(num_collocation_points, 1) * (wavelengths_tensor.max() - wavelengths_tensor.min()) + wavelengths_tensor.min()
    lambda_norm_collocation = 2 * (lambda_collocation - wavelengths_tensor.min()) / (wavelengths_tensor.max() - wavelengths_tensor.min()) - 1

    e_up_colloc, e_down_colloc = network(z_collocation, lambda_norm_collocation).unbind(dim=1)

    grad_e_up = torch.autograd.grad(e_up_colloc, z_collocation, grad_outputs=torch.ones_like(e_up_colloc), create_graph=True)[0]
    grad_e_down = torch.autograd.grad(e_down_colloc, z_collocation, grad_outputs=torch.ones_like(e_down_colloc), create_graph=True)[0]

    k = k_func(lambda_collocation, LAI_learnable)
    s = s_func(lambda_collocation, LAI_learnable)

    residuo1 = grad_e_up - (-k * e_up_colloc + s * e_down_colloc)
    residuo2 = grad_e_down - (k * e_down_colloc - s * e_up_colloc)
    loss_pde = torch.mean(residuo1**2 + residuo2**2)

    # 2. Perda nas Condições de Contorno
    num_data_points = R_umida_tensor.shape[0]
    z_top = H * torch.ones(num_data_points, 1, requires_grad=True)
    z_base = torch.zeros(num_data_points, 1, requires_grad=True)
    lambda_bc = wavelengths_tensor.reshape(-1, 1)
    lambda_norm_bc = 2 * (lambda_bc - wavelengths_tensor.min()) / (wavelengths_tensor.max() - wavelengths_tensor.min()) - 1

    e_up_top, e_down_top = network(z_top, lambda_norm_bc).unbind(dim=1)
    e_up_base, e_down_base = network(z_base, lambda_norm_bc).unbind(dim=1)

    loss_bc_top = torch.mean((e_down_top - 1.0)**2)
    loss_bc_base = torch.mean((e_up_base - R_solo_tensor.squeeze() * e_down_base)**2)
    loss_bc = loss_bc_top + loss_bc_base

    # 3. Perda nos Dados de Medição
    reflectance_pred = e_up_top / (e_down_top + 1e-8)
    loss_data = torch.mean((reflectance_pred - R_umida_tensor.squeeze())**2)

    total_loss = w_pde*loss_pde + w_bc*loss_bc + w_data*loss_data
    return total_loss

def train_pinn(pinn_model, LAI_initial_guess, R_solo, R_umida, wavelengths, H, k_func, s_func, num_epochs=5000):
    """
    Executa o loop de treinamento para a PINN.
    """
    LAI = nn.Parameter(torch.tensor([LAI_initial_guess], requires_grad=True))
    R_umida_tensor = torch.tensor(R_umida, dtype=torch.float32)
    R_solo_tensor = torch.tensor(R_solo, dtype=torch.float32)
    wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32)

    optimizer = torch.optim.Adam(list(pinn_model.parameters()) + [LAI], lr=0.001)
    loss_history = []

    print(f"\nIniciando treinamento da PINN por {num_epochs} épocas...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = calculate_loss(pinn_model, LAI, R_solo_tensor, R_umida_tensor, wavelengths_tensor, H, k_func, s_func)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, LAI: {LAI.item():.4f}")

    print("\nTreinamento concluído.")
    final_lai = LAI.item()
    print(f"LAI final recuperado: {final_lai:.4f}")
    return final_lai, loss_history
