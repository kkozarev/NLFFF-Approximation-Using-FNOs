import torch

class EvaluationMetrics:
    def __init__(self, B_ref, B_ext):
        """
        Initializes the MagneticFieldMetrics class.

        Args:
            B_ref (torch.Tensor): The reference magnetic field.
                                  Expected shape: (3, Nx, Ny, Nz), where the first
                                  dimension holds the Bx, By, and Bz components.
            B_ext (torch.Tensor): The extrapolated magnetic field.
                                  Expected shape: (3, Nx, Ny, Nz).
        """
        self.device = B_ref.device

        bx_ext, by_ext, bz_ext = B_ext[0], B_ext[1], B_ext[2]
        
        self.div_B_ext_grid = self.divergence(bx_ext, by_ext, bz_ext)
        
        jx, jy, jz = self.curl(bx_ext, by_ext, bz_ext)
        self.J_grid = torch.stack([jx, jy, jz], dim=0)

        # Reshape from (3, Nx, Ny, Nz) to (M, 3) where M = Nx*Ny*Nz
        self.B_ref = torch.moveaxis(B_ref, 0, -1).reshape(-1, 3)
        self.B_ext = torch.moveaxis(B_ext, 0, -1).reshape(-1, 3)
        self.J = torch.moveaxis(self.J_grid, 0, -1).reshape(-1, 3)
        self.div_B_ext = self.div_B_ext_grid.flatten()

        self.M = self.B_ref.shape[0]  # Total number of grid points

    @staticmethod
    def divergence(bx, by, bz):
        """Calculates the divergence of a vector field B = (bx, by, bz)."""
        grad_bx_dx = torch.gradient(bx, dim=0)[0]
        grad_by_dy = torch.gradient(by, dim=1)[0]
        grad_bz_dz = torch.gradient(bz, dim=2)[0]
        divergence = grad_bx_dx + grad_by_dy + grad_bz_dz

        return divergence

    @staticmethod
    def curl(bx, by, bz):
        """Calculates the curl of a vector field B = (bx, by, bz)."""
        dbz_dy = torch.gradient(bz, dim=1)[0]
        dby_dz = torch.gradient(by, dim=2)[0]
        dbx_dz = torch.gradient(bx, dim=2)[0]
        dbz_dx = torch.gradient(bz, dim=0)[0]
        dby_dx = torch.gradient(by, dim=0)[0]
        dbx_dy = torch.gradient(bx, dim=1)[0]
        jx = dbz_dy - dby_dz
        jy = dbx_dz - dbz_dx
        jz = dby_dx - dbx_dy
        return jx, jy, jz

    def calculate_C_vec(self):
        """Calculates the vector correlation coefficient (C_vec)."""
        numerator = torch.sum(self.B_ref * self.B_ext)
        denominator = torch.sqrt(torch.sum(self.B_ref**2) * torch.sum(self.B_ext**2))
        result_tensor = numerator / denominator if denominator > 0 else torch.tensor(0.0, device=self.device)
        return result_tensor.item()

    def calculate_C_CS(self):
        """Calculates the cosine similarity index (C_CS)."""
        dot_product = torch.sum(self.B_ref * self.B_ext, dim=1)
        norm_product = torch.linalg.norm(self.B_ref, dim=1) * torch.linalg.norm(self.B_ext, dim=1)
        
        valid_indices = norm_product > 1e-9
        if not torch.any(valid_indices):
            return 0.0
        
        result_tensor = torch.mean(dot_product[valid_indices] / norm_product[valid_indices])
        return result_tensor.item()

    def calculate_E_n(self):
        """Calculates the mean error normalized by the average vector norm (E_n)."""
        numerator = torch.sum(torch.linalg.norm(self.B_ext - self.B_ref, dim=1))
        denominator = torch.sum(torch.linalg.norm(self.B_ref, dim=1))
        result_tensor = numerator / denominator if denominator > 0 else torch.tensor(0.0, device=self.device)

        return result_tensor.item()

    def calculate_E_m(self):
        """Calculates the mean error normalized per vector (E_m)."""
        diff_norm = torch.linalg.norm(self.B_ext - self.B_ref, dim=1)
        ref_norm = torch.linalg.norm(self.B_ref, dim=1)

        valid_indices = ref_norm > 1e-9
        if not torch.any(valid_indices):
            return 0.0
        
        result_tensor = torch.mean(diff_norm[valid_indices] / ref_norm[valid_indices])
        return result_tensor.item()

    def calculate_E_prime_m_n(self):
        """Calculates the transformed error metrics (E'_m and E'_n)."""
        E_n = self.calculate_E_n()
        E_m = self.calculate_E_m()
        return 1.0 - E_n, 1.0 - E_m

    def calculate_epsilon(self):
        """Calculates the relative magnetic energy (epsilon)."""
        numerator = torch.sum((self.B_ext)**2)
        denominator = torch.sum(self.B_ref**2)
        if denominator == 0 and numerator == 0:
            return 1.0
        
        result_tensor = numerator / denominator
        return result_tensor.item()

    def calculate_L_div_n(self):
        """Calculates the normalized divergence (L_div,n) to quantify divergence-freeness."""
        norm_B_ext = torch.linalg.norm(self.B_ext, dim=1)

        result_tensor = torch.mean(torch.abs(self.div_B_ext) / (norm_B_ext + 1e-8))
        return result_tensor.item()

    def calculate_sigma_J(self):
        """Calculates the force-freeness metric (sigma_J)."""
        cross_product_norm = torch.linalg.norm(torch.linalg.cross(self.J, self.B_ext), dim=1)
        norm_B_ext = torch.linalg.norm(self.B_ext, dim=1)
        norm_J = torch.linalg.norm(self.J, dim=1)

        denominator_sum_J = torch.sum(norm_J)
        if denominator_sum_J == 0:
            return 0.0

        valid_indices = norm_B_ext > 1e-9
        if not torch.any(valid_indices):
            return 0.0

        numerator = torch.sum(cross_product_norm[valid_indices] / norm_B_ext[valid_indices])
        result_tensor = numerator / denominator_sum_J
        return result_tensor.item()
    
    def compute_all_metrics(self):
        """Computes all evaluation metrics and returns them in a dictionary."""
        metrics = {
            'C_vec': self.calculate_C_vec(),
            'C_CS': self.calculate_C_CS(),
            "E'_n": self.calculate_E_prime_m_n()[0],
            "E'_m": self.calculate_E_prime_m_n()[1],
            'epsilon': self.calculate_epsilon(),
            'L_div_n': self.calculate_L_div_n(),
            'sigma_J': self.calculate_sigma_J()
        }
        return metrics

def evaluate_magnetic_fields(B_ref, B_ext):
    """
    Evaluates the extrapolated magnetic field against the reference field using various metrics.

    Args:
        B_ref (torch.Tensor): The reference magnetic field.
                              Expected shape: (3, Nx, Ny, Nz), where the first
                              dimension holds the Bx, By, and Bz components.
        B_ext (torch.Tensor): The extrapolated magnetic field.
                              Expected shape: (3, Nx, Ny, Nz).

    Returns:
        dict: A dictionary containing the computed evaluation metrics.
    """
    metrics_calculator = EvaluationMetrics(B_ref, B_ext)
    return metrics_calculator.compute_all_metrics()