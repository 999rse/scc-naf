import numpy as np
from .base import AttentionForest
from .forests import FORESTS, ForestKind, ForestType, TaskType
from typing import Optional, Tuple, Union, Callable
from dataclasses import InitVar, dataclass, field
import logging
from time import time
from numba import njit
import torch
from torch.utils.data import DataLoader, TensorDataset
from .naf_nn import NAFNetwork
from sklearn.utils.validation import check_random_state
from tqdm import tqdm


@njit
def _prepare_leaf_sparse_batch(xs, leaf_ids):
    """
    Args:
        xs: Input data of shape (n_samples, n_features).
        leaf_ids: Leaf id for each sample and tree, of shape (n_samples, n_trees)
    Returns:
        Array of shape (n_samples, n_trees, n_leaves).
    """
    # leaf_ids shape: (n_samples, n_trees)
    max_leaf_id = leaf_ids.max()
    n_leaves = max_leaf_id + 1
    n_trees = leaf_ids.shape[1]
    n_samples = xs.shape[0]
    result = np.zeros((n_samples, n_trees, n_leaves), dtype=np.uint8)
    for i in range(n_samples):
        for j in range(n_trees):
            result[i, j, leaf_ids[i, j]] = 1
    return result


def _prepare_leaf_sparse(xs, leaf_ids, batch_size):
    """
    Args:
        xs: Input data of shape (n_samples, n_features).
        leaf_ids: Leaf id for each sample and tree, of shape (n_samples, n_trees)
        batch_size: Size of each batch to process.
    Returns:
        Array of shape (n_samples, n_trees, n_leaves).
    """
    # Calculate number of batches
    numB = int(xs.shape[0] / batch_size)

    # Initialize result array
    max_leaf_id = leaf_ids.max()
    n_leaves = max_leaf_id + 1
    n_trees = leaf_ids.shape[1]
    result = np.zeros((numB * batch_size, n_trees, n_leaves), dtype=np.uint8)

    # Iterate over batches
    for i in tqdm(range(0, numB * batch_size, batch_size)):
        batch_xs = xs[i:i+batch_size]
        batch_leaf_ids = leaf_ids[i:i+batch_size]

        # Generate leaf sparse representation for current batch
        batch_result = _prepare_leaf_sparse_batch(batch_xs, batch_leaf_ids)
        batch_result = np.resize(batch_result, (batch_size, n_trees, n_leaves))
        # Update result array with current batch
        result[i:i+batch_size] = batch_result
    return result

@dataclass
class NAFParams:
    """Parameters of Neural Attention Forest."""
    kind: Union[ForestKind, str]
    task: TaskType
    loss: Union[str, Callable] = 'mse'
    eps: Optional[int] = None
    mode: str = 'end_to_end'
    n_epochs: int = 100
    lr: float = 1.e-3
    lam: float = 0.0
    hidden_size: int = 16
    n_layers: int = 1
    target_loss_weight: float = 1.0
    forest: dict = field(default_factory=lambda:{})
    use_weights_random_init: bool = True
    weights_init_type: str = 'default'
    random_state: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.kind, ForestKind):
            self.kind = ForestKind.from_name(self.kind)


class NeuralAttentionForest(AttentionForest):
    def __init__(self, params: NAFParams, run_agent=None, device='cpu', batch_size=None):
        self.params = params
        self.forest = None
        self.run_agent = run_agent
        self._after_init()
        self.device = device
        self.batch_size = batch_size

    def _make_nn(self, n_features):
        self.nn = NAFNetwork(n_features, self.params.hidden_size, self.params.n_layers)
        if self.params.use_weights_random_init:
            MAX_INT = np.iinfo(np.int32).max
            rng = check_random_state(self.params.random_state)
            seed = rng.randint(MAX_INT)
            torch.manual_seed(seed)
            def _init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    # torch.nn.init.uniform_(m.weight)
                    if self.params.weights_init_type == 'xavier':
                        torch.nn.init.xavier_normal_(m.weight)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'uniform':
                        torch.nn.init.uniform_(m.weight)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'general_rule_uniform':
                        n = m.in_features
                        y = 1.0 / np.sqrt(n)
                        m.weight.data.uniform_(-y, y)
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'general_rule_normal':
                        y = m.in_features
                        m.weight.data.normal_(0.0, 1.0 / np.sqrt(y))
                        m.bias.data.fill_(0.0)
                    elif self.params.weights_init_type == 'default':
                        m.reset_parameters()
                    else:
                        raise ValueError(f'Wrong {self.params.weights_init_type=}')
            self.nn.apply(_init_weights)

    def _base_fit(self, X, y) -> 'NeuralAttentionForest':
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        self.forest = forest_cls(**self.params.forest)
        self.forest.random_state = self.params.random_state

        logging.debug("Start fitting Random forest")
        start_time = time()
        print('Start fitting Random forest')
        self.forest.fit(X, y)
        # sumL, sumA, numB = 0, 0, int(len(X)/self.batch_size)
        # for i in tqdm(range(0, numB*self.batch_size, self.batch_size)):
        #     X_b = X[i:i+self.batch_size]
        #     y_b = y[i:i+self.batch_size]
        #     self.forest.fit(X_b, y_b)
        end_time = time()
        logging.info("Random forest fit time: %f", end_time - start_time)
        # store training X and y
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        start_time = time()
        print('Start random forest apply')
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        end_time = time()
        logging.info("Random forest apply time: %f", end_time - start_time)
        # make a tree-leaf-points correspondence
        logging.debug("Generating leaves data")
        start_time = time()
        print('Start leaf generate')
        self.leaf_sparse = _prepare_leaf_sparse(self.training_xs, self.training_leaf_ids, self.batch_size)
        end_time = time()
        logging.info("Leaf generation time: %f", end_time - start_time)
        # self.tree_weights = np.ones(self.forest.n_estimators)
        logging.debug("Initializing the neural network")
        print('Initializing the neural network')
        self.n_trees = self.forest.n_estimators
        self._make_nn(n_features=X.shape[1])
        return self

    def fit(self, x, y):
        self._base_fit(x, y)

    def optimize_weights(self, X, y_orig) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        if self.params.mode == 'end_to_end':
            self._optimize_weights_end_to_end(X, y_orig)
        elif self.params.mode == 'two_step':
            self._optimize_weights_two_step(X, y_orig)
        else:
            raise ValueError(f'Wrong mode: {self.params.mode!r}')

    def _make_loss(self):
        if callable(self.params.loss):
            return self.params.loss
        elif self.params.loss == 'mse':
            return torch.nn.MSELoss()
        raise ValueError(f'Wrong loss: {self.params.loss!r}')

    # ГПУ + ОПТИМИЗАЦИЯ ПО БАТЧАМ
    def _optimize_weights_end_to_end(self, X, y_orig) -> 'NeuralAttentionForest':
        print(self.device)
        assert self.forest is not None, "Need to fit before weights optimization"
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=True)
        X_tensor = torch.tensor(X, dtype=torch.double) #.to(self.device)
        background_X = torch.tensor(self.training_xs, dtype=torch.double) #.to(self.device)
        background_y = torch.tensor(self.training_y, dtype=torch.double) #.to(self.device)
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
            y_orig = y_orig[:, np.newaxis]
        y_true = torch.tensor(y_orig, dtype=torch.double) #.to(self.device)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool) #.to(self.device)

        self.nn = self.nn.to(self.device)
        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs
        
        if self.params.lam == 0.0:
            for epoch in range(n_epochs):
                predictions = self.nn(
                    X_tensor,
                    background_X,
                    background_y,
                    neighbors_hot,
                )
                optim.zero_grad()
                loss = loss_fn(predictions, y_true)
                loss.backward()
                optim.step()
        else:  # self.params.lam > 0.0
            tlw = self.params.target_loss_weight
            lam = self.params.lam
            for epoch in tqdm(range(n_epochs)):
                numB = int(len(X)/self.batch_size)
                for i in range(0, numB*self.batch_size, self.batch_size):
                    X_b = (background_X[i:i+self.batch_size]).to(self.device)
                    y_b = (background_y[i:i+self.batch_size]).to(self.device)
                    X_tensor_b = (X_tensor[i:i+self.batch_size]).to(self.device)
                    nei_hot_b = (neighbors_hot[i:i+self.batch_size, i:i+self.batch_size]).to(self.device)
                    predictions, xs_reconstruction, _alphas, _betas = self.nn(
                        X_tensor_b,
                        X_b,
                        y_b,
                        nei_hot_b,
                        need_attention_weights=True,
                    )
                    y_true_b = y_true[i:i+self.batch_size]
                    optim.zero_grad()
                    loss = tlw * loss_fn(predictions, y_true_b.to(self.device)) + lam * loss_fn(xs_reconstruction, X_tensor_b)
                    loss.backward()
                    optim.step()
        return self.nn.cpu()

    # ГПУ
    # def _optimize_weights_end_to_end(self, X, y_orig) -> 'NeuralAttentionForest':
    #     print(self.device)
    #     assert self.forest is not None, "Need to fit before weights optimization"
    #     neighbors_hot = self._get_leaf_data_segments(X, exclude_input=True)
    #     X_tensor = torch.tensor(X, dtype=torch.double).to(self.device)
    #     background_X = torch.tensor(self.training_xs, dtype=torch.double).to(self.device)
    #     background_y = torch.tensor(self.training_y, dtype=torch.double).to(self.device)
    #     if len(background_y.shape) == 1:
    #         background_y = background_y.unsqueeze(1)
    #         y_orig = y_orig[:, np.newaxis]
    #     y_true = torch.tensor(y_orig, dtype=torch.double).to(self.device)
    #     neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool).to(self.device)
    #     self.nn = self.nn.to(self.device)

    #     optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
    #     loss_fn = self._make_loss()
    #     n_epochs = self.params.n_epochs

    #     if self.params.lam == 0.0:
    #         for epoch in range(n_epochs):
    #             predictions = self.nn(
    #                 X_tensor,
    #                 background_X,
    #                 background_y,
    #                 neighbors_hot,
    #             )
    #             optim.zero_grad()
    #             loss = loss_fn(predictions, y_true)
    #             loss.backward()
    #             optim.step()
    #     else:  # self.params.lam > 0.0
    #         tlw = self.params.target_loss_weight
    #         lam = self.params.lam
    #         for epoch in range(n_epochs):
    #             # second_y, second_xs, first_alphas, second_betas
    #             predictions, xs_reconstruction, _alphas, _betas = self.nn(
    #                 X_tensor,
    #                 background_X,
    #                 background_y,
    #                 neighbors_hot,
    #                 need_attention_weights=True,
    #             )
    #             optim.zero_grad()
    #             loss = tlw * loss_fn(predictions, y_true) + lam * loss_fn(xs_reconstruction, X_tensor)
    #             loss.backward()
    #             optim.step()
    #     return self.nn.cpu()

    def _optimize_weights_two_step(self, X, y_orig) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=True)
        X_tensor = torch.tensor(X, dtype=torch.double)
        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y, dtype=torch.double)
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
            y_orig = y_orig[:, np.newaxis]
        y_true = torch.tensor(y_orig, dtype=torch.double)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)

        # first step
        first_nn = self.nn.leaf_network
        optim = torch.optim.AdamW(first_nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs
        n_trees = neighbors_hot.shape[2]
        y_true_per_tree = y_true[:, None].repeat(1, n_trees, 1)
        n_out = y_true_per_tree.shape[-1]
        for epoch in range(n_epochs // 2):
            _first_leaf_xs, first_leaf_y, _first_alphas = first_nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
            )
            # first_leaf_y shape: (n_samples, n_trees, n_out)
            optim.zero_grad()
            loss = loss_fn(first_leaf_y.view(-1, n_out), y_true_per_tree.view(-1, n_out))
            loss.backward()
            optim.step()

        self.nn.tree_network.second_encoder.weight.data[:] = first_nn.first_encoder.weight.data
        self.nn.tree_network.second_encoder.bias.data[:] = first_nn.first_encoder.bias.data
        # second step
        optim = torch.optim.AdamW(self.nn.tree_network.parameters(), lr=self.params.lr)
        loss_fn = torch.nn.MSELoss()
        for epoch in range(n_epochs // 2):
            predictions = self.nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
            )
            optim.zero_grad()
            loss = loss_fn(predictions, y_true)
            loss.backward()
            optim.step()

        return self

    def optimize_weights_unlabeled(self, X) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        if self.params.mode == 'end_to_end':
            self._optimize_weights_unlabeled_end_to_end(X)
        else:
            raise ValueError(f'Wrong mode: {self.params.mode!r}')

    def _optimize_weights_unlabeled_end_to_end(self, X) -> 'NeuralAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=False)
        X_tensor = torch.tensor(X, dtype=torch.double)
        background_X = torch.tensor(self.training_xs, dtype=torch.double)
        background_y = torch.tensor(self.training_y, dtype=torch.double)
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool)

        optim = torch.optim.AdamW(self.nn.parameters(), lr=self.params.lr)
        loss_fn = self._make_loss()
        n_epochs = self.params.n_epochs

        for epoch in range(n_epochs):
            # second_y, second_xs, first_alphas, second_betas
            predictions, xs_reconstruction, _alphas, _betas = self.nn(
                X_tensor,
                background_X,
                background_y,
                neighbors_hot,
                need_attention_weights=True,
            )
            optim.zero_grad()
            loss = loss_fn(xs_reconstruction, X_tensor)
            loss.backward()
            optim.step()
        return self


    def _get_leaf_data_segments(self, X, exclude_input=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            X: Input points.
            exclude_input: Exclude leaf points that are exactly the same as input point.
                           It is useful to unbias training when fitting and optimizing
                           on the same data set.
        """
        leaf_ids = self.forest.apply(X)
        # shape of leaf_ids: (n_samples, n_trees)
        result = np.zeros((X.shape[0], self.leaf_sparse.shape[0], self.leaf_sparse.shape[1]), dtype=np.uint8)
        # shape of `self.leaf_sparse`: (n_background_samples, n_trees, n_leaves)
        for i in range(leaf_ids.shape[0]):
            for j in range(leaf_ids.shape[1]):
                result[i, :, j] = self.leaf_sparse[:, j, leaf_ids[i, j]]
            if exclude_input:
                result[i, i, :] = 0
        # result shape: (n_samples, n_background_samples, n_trees)
        return result

    def predict(self, X, need_attention_weights=False) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        # all_leaf_x, all_leaf_y, sample_ids, tree_ids = self._get_leaf_data_segments(X, exclude_input=False)
        neighbors_hot = self._get_leaf_data_segments(X, exclude_input=False)
        X_tensor = torch.tensor(X, dtype=torch.double) #.to(self.device)
        background_X = torch.tensor(self.training_xs, dtype=torch.double) #.to(self.device)
        background_y = torch.tensor(self.training_y, dtype=torch.double) #.to(self.device)
        if len(background_y.shape) == 1:
            background_y = background_y.unsqueeze(1)
        neighbors_hot = torch.tensor(neighbors_hot, dtype=torch.bool) #.to(self.device)
        self.nn = self.nn.to(self.device)
        print('Start predict')
        with torch.no_grad():
            all_predictions = []
            all_X_reconstruction = []
            all_alphas = []
            all_betas = []
            numB = int(len(X)/self.batch_size)
            for i in tqdm(range(0, numB*self.batch_size, self.batch_size)):
                X_b = (background_X[i:i+self.batch_size]).to(self.device)
                y_b = (background_y[i:i+self.batch_size]).to(self.device)
                X_tensor_b = (X_tensor[i:i+self.batch_size]).to(self.device)
                nei_hot_b = (neighbors_hot[i:i+self.batch_size, i:i+self.batch_size]).to(self.device)
                output = self.nn(
                    X_tensor_b,
                    X_b,
                    y_b,
                    nei_hot_b,
                    need_attention_weights=need_attention_weights,
                )
                if isinstance(output, tuple):
                    output = tuple([
                        out.detach().cpu().numpy()
                        for out in output
                    ])
                    predictions, X_reconstruction, alphas, betas = output
                    all_X_reconstruction = np.append(all_X_reconstruction, X_reconstruction)
                    all_alphas = np.append(all_alphas, alphas)
                    all_betas = np.append(all_betas, betas)
                else:
                    predictions = output.detach().cpu().numpy()
                
                all_predictions = np.append(all_predictions, predictions)

        if self.params.kind.need_add_init():
            all_predictions += self.forest.init_.predict(X)[:, np.newaxis]
        if not need_attention_weights:
            return all_predictions
        else:
            return all_predictions, all_X_reconstruction, all_alphas, all_betas
