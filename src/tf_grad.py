import torch
import torch.optim as optim


class TensorFactorization:
    def __init__(self, tensor, rank, method="cp", mask=None, constraint=None):
        """
        Initialize a tensor decomposition class supporting CP, Tucker, Tensor Train, and Ring decompositions.

        Args:
        - tensor: torch.Tensor, the tensor to decompose.
        - rank: int or tuple, the target rank for decomposition.
        - method: str, the decomposition method ("cp", "tucker", "train", "ring").
        - mask: torch.Tensor, binary mask of the same shape as tensor (1 for observed, 0 for missing).
        - constraint: torch.Tensor, tensor representing constraint satisfaction (same shape as tensor).
        """
        if mask is None:
            mask = torch.ones_like(tensor)
        if constraint is None:
            constraint = torch.ones_like(tensor)
        assert tensor.shape == mask.shape == constraint.shape, "Tensor, mask, and constraint must have the same shape."

        self.tensor = tensor
        self.mask = mask
        self.constraint = constraint
        self.method = method.lower()

        if self.method == "cp":
            self.rank = rank
            self.dims = tensor.shape
            self.factors = [torch.randn(dim, rank, requires_grad=True) for dim in self.dims]

        elif self.method == "tucker":
            self.rank = rank if isinstance(rank, tuple) else (rank,) * len(tensor.shape)
            self.core = torch.randn(*self.rank, requires_grad=True)
            self.factors = [torch.randn(dim, r, requires_grad=True) for dim, r in zip(tensor.shape, self.rank)]

        elif self.method == "train":
            self.ranks = rank if isinstance(rank, list) else [rank] * (len(tensor.shape) + 1)
            assert self.ranks[0] == self.ranks[-1] == 1, "Tensor Train ranks must start and end with 1."
            self.factors = [
                torch.randn(self.ranks[i], tensor.shape[i], self.ranks[i + 1], requires_grad=True)
                for i in range(len(tensor.shape))
            ]

        elif self.method == "ring":
            self.rank = rank
            self.factors = [
                torch.randn(rank, tensor.shape[i], rank, requires_grad=True)
                for i in range(len(tensor.shape))
            ]

        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'cp', 'tucker', 'train', or 'ring'.")
        
        print(f"Initialized {method} decomposition with rank {rank}.")

    def reconstruct(self):
        """
        Reconstruct the tensor based on the decomposition method.
        """
        if self.method == "cp":
            R = self.rank
            recon = torch.zeros_like(self.tensor)
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
            # Get number of modes
            n_modes = len(self.factors)
            
            # Start with first core
            result = self.factors[0]  # [r, d1, r]
            
            # Contract with middle cores
            for i in range(1, n_modes-1):
                # Contract [r,di,r] with [r,d(i+1),r] -> [r,di,d(i+1),r]
                result = torch.einsum('ijk,klm->ijlm', result, self.factors[i])
                # Reshape to [r,di*d(i+1),r] for next iteration
                s1, s2, s3, s4 = result.shape
                result = result.reshape(s1, s2*s3, s4)
                
            # Final contraction with last core to close the ring
            # Contract [r,d1*...*d(n-1),r] with [r,dn,r] -> [d1*...*d(n-1),dn]
            result = torch.einsum('ijk,klm->jl', result, self.factors[-1])
            
            # Reshape to match original tensor dimensions
            result = result.reshape(self.tensor.shape)
            
            return result

    def optimize(self, lr=0.01, max_iter=1000, tol=1e-6, reg_lambda=0.01, constraint_lambda=1):
        """
        Perform optimization for the specified decomposition method.

        Args:
        - lr: float, learning rate.
        - max_iter: int, maximum number of iterations.
        - tol: float, tolerance for convergence.
        - reg_lambda: float, regularization coefficient for L2 regularization.
        - constraint_lambda: float, penalty coefficient for constraint violations.

        Returns:
        - factors: Optimized factor matrices or tensors for the decomposition method.
        """
        params = self.factors if self.method != "tucker" else [self.core] + self.factors
        optimizer = optim.Adam(params, lr=lr)
        prev_loss = float('inf')

        for iteration in range(max_iter):
            optimizer.zero_grad()

            # Reconstruct the tensor
            reconstruction = self.reconstruct()

            # Compute the loss function
            def loss_fn():
                # Reconstruction error on observed entries weighted by constraints
                error_term = self.constraint * self.mask * (self.tensor - reconstruction)
                mse_loss = torch.norm(error_term) ** 2

                # Constraint violation penalty
                violation_term = torch.clamp((1 - self.constraint) * reconstruction, min=0)
                constraint_loss = constraint_lambda * torch.sum(violation_term)

                # Regularization term
                l2_loss = reg_lambda * sum(torch.norm(factor) ** 2 for factor in params)

                # Total loss
                total_loss = mse_loss + constraint_loss + l2_loss
                return total_loss, mse_loss, constraint_loss, l2_loss

            # Compute the loss
            loss, mse_loss, constraint_loss, l2_loss = loss_fn()
            loss.backward()
            optimizer.step()

            # Monitor loss
            msg1 = f"Iteration {iteration + 1}, Loss: {loss.item():.6f}"
            msg2 = f"mse: {mse_loss.item():.6f}, constraint: {constraint_loss.item():.6f}, l2: {l2_loss.item():.6f}"
            if iteration == max_iter - 1:
                logging.info(msg1)
                logging.info(msg2)
            else:
                print(msg1)
                print(msg2)
            if abs(prev_loss - loss.item()) < tol:
                print("Converged.")
                break
            prev_loss = loss.item()

        return [factor.detach() for factor in params]



if __name__ == "__main__":
    import time
    import logging
    from datetime import datetime

    torch.manual_seed(42)

    def run_tf(dim, mode, rank, method):
        dim = (dim,)
        shape = dim * mode
        tensor = torch.randn(shape)
        constraint = (torch.rand_like(tensor) > 0.5).float()

        if method == "cp":
            rank = rank
        elif method == "tucker":
            rank = (rank,) * mode
        elif method == "train":
            rank = [1] + [rank] * (mode - 1) + [1]
        elif method == "ring":
            rank = rank

        tf = TensorFactorization(tensor, rank=rank, method=method, mask=None, constraint=constraint)
        factors = tf.optimize()
        reconstruction = tf.reconstruct()

    dim = 7
    rank = 3
    methods = ["cp", "tucker", "train", "ring"]
    modes = [2, 3, 4, 5, 6, 7, 8, 9]


    # 設定: ログファイル名とフォーマット
    log_filename = f"tensor_factorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # ログにヘッダーを記録
    logging.info("Starting tensor factorization benchmarking")
    logging.info(f"Dimensions: {dim}")
    logging.info(f"Methods: {methods}")
    logging.info(f"Modes tested: {modes}")
    logging.info("-----------------------------------------")

    # モードごとに時間を計測してログ出力
    def benchmark_tf():
        for m in modes:
            for method in methods:
                logging.info(f"Testing mode: {m}, method: {method}")

                # 時間計測を開始
                start_time = time.time()

                try:
                    # 分解を実行
                    run_tf(dim=dim, mode=m, rank=rank, method=method)

                    # 計測終了
                    elapsed_time = time.time() - start_time

                    logging.info(f"Mode: {m}, Method: {method}, Time: {elapsed_time:.6f} seconds")

                except Exception as e:
                    # エラーが発生した場合、ログに記録
                    logging.error(f"Error occurred for mode: {m}, method: {method}. Exception: {e}")

        logging.info("Benchmarking completed.")


    benchmark_tf()
    print(f"Benchmarking completed. Results are saved in {log_filename}.")



# if __name__ == "__main__":
#     # Example Usage
#     torch.manual_seed(42)
#     I, J, K = 10, 8, 6
#     rank = 3
#     tensor = torch.randn(I, J, K)
#     mask = (torch.rand_like(tensor) > 0.2).float()
#     constraint = (torch.rand_like(tensor) > 0.5).float()

#     # Perform CP decomposition
#     cp_decomp = TensorFactorization(tensor, rank=rank, method="cp", mask=mask, constraint=constraint)
#     factors_cp = cp_decomp.optimize()

#     # Perform Tucker decomposition
#     tucker_decomp = TensorFactorization(tensor, rank=(3, 3, 3), method="tucker", mask=mask, constraint=constraint)
#     factors_tucker = tucker_decomp.optimize()

#     # Perform Tensor Train decomposition
#     tt_decomp = TensorFactorization(tensor, rank=[1, 3, 3, 1], method="train", mask=mask, constraint=constraint)
#     factors_tt = tt_decomp.optimize()

#     # Perform Ring decomposition
#     ring_decomp = TensorFactorization(tensor, rank=3, method="ring", mask=mask, constraint=constraint)
#     factors_ring = ring_decomp.optimize()