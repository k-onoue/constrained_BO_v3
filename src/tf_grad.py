import torch
import torch.optim as optim


class TensorFactorization:
    def __init__(self, tensor, rank, method="cp", mask=None, constraint=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Move tensors to the device
        tensor = tensor.to(self.device)
        if mask is None:
            mask = torch.ones_like(tensor, device=self.device)
        else:
            mask = mask.to(self.device)
        if constraint is None:
            constraint = torch.ones_like(tensor, device=self.device)
        else:
            constraint = constraint.to(self.device)

        assert tensor.shape == mask.shape == constraint.shape, "Tensor, mask, and constraint must have the same shape."

        self.tensor = tensor
        self.mask = mask
        self.constraint = constraint
        self.method = method.lower()

        self.total_params = 0  # Initialize total_params

        if self.method == "cp":
            self.rank = rank
            self.dims = tensor.shape
            self.factors = [torch.randn(dim, rank, requires_grad=True, device=self.device) for dim in self.dims]
            self.total_params = sum(factor.numel() for factor in self.factors)

        elif self.method == "tucker":
            self.rank = rank if isinstance(rank, tuple) else (rank,) * len(tensor.shape)
            self.core = torch.randn(*self.rank, requires_grad=True, device=self.device)
            self.factors = [torch.randn(dim, r, requires_grad=True, device=self.device) for dim, r in zip(tensor.shape, self.rank)]
            self.total_params = self.core.numel() + sum(factor.numel() for factor in self.factors)

        elif self.method == "train":
            self.ranks = rank if isinstance(rank, list) else [rank] * (len(tensor.shape) + 1)
            assert self.ranks[0] == self.ranks[-1] == 1, "Tensor Train ranks must start and end with 1."
            self.factors = [
                torch.randn(self.ranks[i], tensor.shape[i], self.ranks[i + 1], requires_grad=True, device=self.device)
                for i in range(len(tensor.shape))
            ]
            self.total_params = sum(factor.numel() for factor in self.factors)

        elif self.method == "ring":
            self.rank = rank
            self.factors = [
                torch.randn(rank, tensor.shape[i], rank, requires_grad=True, device=self.device)
                for i in range(len(tensor.shape))
            ]
            self.total_params = sum(factor.numel() for factor in self.factors)

        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'cp', 'tucker', 'train', or 'ring'.")

        # for logging
        self.loss = None
        self.mse_loss = None
        self.constraint_loss = None
        self.l2_loss = None
        self.iter_end = None

        print(f"Initialized {method} decomposition with rank {rank} on device {self.device}.")
        print(f"Total parameters: {self.total_params}")

    def reconstruct(self):
        """
        Reconstruct the tensor based on the decomposition method.
        """
        if self.method == "cp":
            R = self.rank
            recon = torch.zeros_like(self.tensor, device=self.device)  # Ensure tensor is on the correct device
            for r in range(R):
                component = torch.ger(self.factors[0][:, r], self.factors[1][:, r])
                for mode in range(2, len(self.dims)):
                    component = component.unsqueeze(-1) * self.factors[mode][:, r]
                recon += component
            return recon

        elif self.method == "tucker":
            recon = self.core
            for i, factor in enumerate(self.factors):
                recon = torch.tensordot(recon, factor, dims=[[0], [1]])
            return recon

        elif self.method == "train":
            recon = self.factors[0]
            for factor in self.factors[1:]:
                recon = torch.einsum("...i,ijk->...jk", recon, factor)
            return recon.squeeze()

        elif self.method == "ring":
            n_modes = len(self.factors)
            result = self.factors[0]
            for i in range(1, n_modes-1):
                result = torch.einsum('ijk,klm->ijlm', result, self.factors[i])
                s1, s2, s3, s4 = result.shape
                result = result.reshape(s1, s2*s3, s4)
            result = torch.einsum('ijk,klm->jl', result, self.factors[-1])
            result = result.reshape(self.tensor.shape)
            return result

    def optimize(self, lr=0.01, max_iter=None, tol=1e-6, reg_lambda=0.01, constraint_lambda=1, verbose=True):
        """
        Perform optimization for the specified decomposition method.

        Args:
        - lr: float, learning rate.
        - max_iter: int or None, maximum number of iterations (if None, use tol for convergence).
        - tol: float, tolerance for convergence.
        - reg_lambda: float, regularization coefficient for L2 regularization.
        - constraint_lambda: float, penalty coefficient for constraint violations.

        Returns:
        - factors: Optimized factor matrices or tensors for the decomposition method.
        """
        params = self.factors if self.method != "tucker" else [self.core] + self.factors
        optimizer = optim.Adam(params, lr=lr)
        prev_loss = float('inf')

        iteration = 0
        while True:
            optimizer.zero_grad()

            reconstruction = self.reconstruct()

            def loss_fn():
                n_se = torch.sum(self.mask)
                n_c = torch.sum(1 - self.constraint)
                n_c = n_c if n_c > 0 else 1
                
                error_term = self.constraint * self.mask * (self.tensor - reconstruction)
                mse_loss = torch.norm(error_term) ** 2 / n_se if n_se > 0 else 0
                
                violation_term = torch.clamp((1 - self.constraint) * (reconstruction - self.tensor), min=0)
                constraint_loss = constraint_lambda * torch.sum(violation_term) / n_c
                
                l2_loss = sum(torch.norm(factor) ** 2 / torch.numel(factor) for factor in params)
                l2_loss *= reg_lambda
                
                total_loss = mse_loss + constraint_loss + l2_loss
                
                return total_loss, mse_loss, constraint_loss, l2_loss

            loss, mse_loss, constraint_loss, l2_loss = loss_fn()
            loss.backward()
            optimizer.step()

            # Logging
            self.loss = loss
            self.mse_loss = mse_loss
            self.constraint_loss = constraint_loss
            self.l2_loss = l2_loss
            
            if verbose:
                print(f"Iter: {iteration}, Loss: {loss}")
                print(f"MSE: {mse_loss}, CONST: {constraint_loss}, L2: {l2_loss}")

            if abs(prev_loss - loss.item()) < tol:
                print("Converged.")
                break
            
            if max_iter is not None and iteration == max_iter - 1:
                break

            prev_loss = loss.item()
            iteration += 1

        self.iter_end = iteration

        return [factor.detach() for factor in params]
