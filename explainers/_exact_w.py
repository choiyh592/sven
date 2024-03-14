import torch
import logging
from tqdm import tqdm

class ExactExplainerW():
    """
    An exact explainer for PyTorch models with feature vectors with various sizes. 
    This explainer calculates exact SHAP values and weighted SHAP values for a given PyTorch model and dataset.
    The explainer is based on the paper "A Unified Approach to Interpreting Model Predictions" by Lundberg and Lee (2017)."
    As the explainer calculates exact SHAP values, it is not suitable for large datasets or models with a large number of features.
    If the dataset is large, consider using ApproximateExplainer1 or ApproximateExplainer2 instead.
    """
    def __init__(self, torchmodel, tensordataset, featurevector_sizes, unitsize = 1, method='regression', num_to_sample=None, nan=0, tolerance = 0.01, device='cuda'):
        """
        Initializes the explainer.
        Args:
            torchmodel (torch.nn.Module): The PyTorch model to explain.
            tensordataset (torch.utils.data.TensorDataset): The dataset to explain.
            featurevector_size (int): The size of the feature vector.
            method (str): The method of the model. Must be one of 'binary_classification_0', 'binary_classification_1', 'regression', 'binary_classification_0_logodds', 'binary_classification_1_logodds'.
            num_to_sample (int): The number of instances to sample from the dataset. If None, samples all instances.
            nan (int): The value to replace NaNs with.
            tolerance (float): The tolerance for the SHAP values to sum to the expected value.
            device (str): The device to use. Must be one of 'cuda', 'cpu'.
        """
        self._model = torchmodel
        self._method = method
        self._featurevector_sizes = featurevector_sizes
        assert method in ['binary_classification_0', 'binary_classification_1', 'regression', 'binary_classification_0_logodds', 'binary_classification_1_logodds'], 'Invalid method'

        # Check for NaNs in the dataset
        if torch.isnan(tensordataset.tensors[0]).any():
            # Warn and convert NaNs to 0
            logging.warning(f'NaN values detected in the dataset. Converting to {nan}.')
            tensordataset.tensors[0][torch.isnan(tensordataset.tensors[0])] = nan
            
        self._dataset = tensordataset
        self._num_instances = len(self._dataset)
        assert torch.sum(self._featurevector_sizes) == len(self._dataset[0][0]), f'The sum of the featurevector_sizes {torch.sum(self._featurevector_sizes)}'\
            f' must be equal to the length of the feature vector {len(self._dataset[0][0])}'
        self._num_features = len(self._featurevector_sizes)
        self._log_of_num_feature_permutations = torch.lgamma(torch.tensor(self._num_features + 1)).item()

        if num_to_sample is None:
            self._num_to_sample = len(self._dataset)
        else:
            assert num_to_sample <= len(self._dataset), 'num_to_sample must be less than or equal to the number of instances in the dataset'
            self._num_to_sample = num_to_sample
                   
        self._sampleddata = self._dataset[torch.randperm(self._num_instances)[:self._num_to_sample]]
        
        if torch.cuda.is_available() & (device == 'cuda'):
            self._device = 'cuda'
        else:
            if device == 'cuda':
                logging.warning('CUDA not available. Using CPU.')
            self._device = 'cpu'

        self._shap_values = torch.zeros((self._num_instances, self._num_features), device=self._device)
        self._true_predictions = self._all_predictions()
        self._shap_expectation = self._true_predictions.mean().item()
        self._calculate_shapley_value_for_all_instances()

        self._shap_summations = torch.sum(self._shap_values, dim=1) + self._shap_expectation

        if torch.max(torch.abs(self._true_predictions - self._shap_summations)) > tolerance:
            logging.warning('SHAP values do not sum to the expected value within the given tolerance. This may be due to numerical instability.'
                            'Consider increasing the tolerance or checking the dataset for NaNs or infinities.')
        
        self._shap_weighted = self._shap_values / (self._featurevector_sizes.to(device) / unitsize)
    
    def _generate_coalition_vectors(self, length, featurevector_sizes):
        if length == 1:
            return torch.tensor([[0] * featurevector_sizes[length - 1], [1] * featurevector_sizes[length - 1]], dtype=torch.float32, device=self._device)
        else:
            previous_combinations = self._generate_coalition_vectors(length - 1, featurevector_sizes)
            zero_appended = torch.cat((previous_combinations, torch.zeros(previous_combinations.shape[0], featurevector_sizes[length - 1], dtype=torch.float32, device=self._device)), dim=1)
            one_appended = torch.cat((previous_combinations, torch.ones(previous_combinations.shape[0], featurevector_sizes[length - 1], dtype=torch.float32, device=self._device)), dim=1)
            return torch.cat((zero_appended, one_appended), dim=0)
    
    def _calculate_shapley_value_for_all_instances(self):
        all_coalition_vectors = self._generate_coalition_vectors(self._num_features, self._featurevector_sizes)
        all_coalition_vectors_rev = 1 - all_coalition_vectors

        for instance_idx, instance in enumerate(tqdm(self._dataset[:][0], position=0, desc='[INIT] Calculating SHAP values for instances')):
            self._calculate_shapley_value_for_instance(instance, all_coalition_vectors, all_coalition_vectors_rev, instance_idx)

    def _calculate_shapley_value_for_instance(self, instance, all_coalition_vectors, all_coalition_vectors_rev, instance_idx):
        all_coalesced_instances = all_coalition_vectors.clone()
        all_coalesced_instances *= instance
        for feature_idx in range(self._num_features):
            self._shap_values[instance_idx, feature_idx] += self._calculate_shapley_value_for_feature(all_coalesced_instances, all_coalition_vectors, all_coalition_vectors_rev, feature_idx)
    
    def _calculate_shapley_value_for_feature(self, all_coalesced_instances, all_coalition_vectors_orig, all_coalition_vectors_rev, feature_index):
        all_coalition_vectors = all_coalition_vectors_orig.clone()
        feature_vector_start_idx = torch.sum(self._featurevector_sizes[:feature_index]).item()
        all_coalition_vectors[:, feature_vector_start_idx : feature_vector_start_idx + self._featurevector_sizes[feature_index]] = 0
        feature_attributions = torch.zeros(all_coalesced_instances.shape[0], device=self._device)
        for coalition_vectors_index, (coalesced_instance, coalition_vector_rev) in enumerate(zip(all_coalesced_instances, all_coalition_vectors_rev)):
            feature_attributions[coalition_vectors_index] += self._calculate_sample_feature_attribution(coalesced_instance, coalition_vector_rev, feature_index, feature_vector_start_idx)

        num_features = self._num_features

        indexes_to_check = torch.cat([torch.sum(self._featurevector_sizes[:i]).unsqueeze(0) for i in range(num_features)], dim=0)
        size_of_coalition_vectors = torch.sum(all_coalition_vectors[:, indexes_to_check], dim=1)
        log_of_size_factorial = torch.lgamma(size_of_coalition_vectors + 1)

        size_of_inv_coalition_vectors_minus_i = -size_of_coalition_vectors + num_features - 1
        log_of_size_inv_factorial = torch.lgamma(size_of_inv_coalition_vectors_minus_i + 1)

        coeffcients_for_feature_attributions = torch.exp(log_of_size_factorial + log_of_size_inv_factorial - self._log_of_num_feature_permutations)

        shapley_value_for_feature = torch.sum(coeffcients_for_feature_attributions * feature_attributions)

        return shapley_value_for_feature
        
    def _calculate_sample_feature_attribution(self, coalesced_instance, coalition_vector_rev, feature_index, feature_vector_start_idx):
        if coalesced_instance[feature_vector_start_idx] == 0:
            return 0
        else:
            sampled_data = self._sampleddata[:][0].clone()
            sampled_data *= coalition_vector_rev
            sampled_data += coalesced_instance

            sampled_data_without_feature = self._sampleddata[:][0].clone()
            coalesced_instance_without_feature = coalesced_instance.clone()
            coalesced_instance_without_feature[feature_vector_start_idx : feature_vector_start_idx + self._featurevector_sizes[feature_index]] = 0
            coalition_vector_rev_without_feature = coalition_vector_rev.clone()
            coalition_vector_rev_without_feature[feature_vector_start_idx : feature_vector_start_idx + self._featurevector_sizes[feature_index]] = 1

            sampled_data_without_feature *= coalition_vector_rev_without_feature
            sampled_data_without_feature += coalesced_instance_without_feature

            return self._predict(sampled_data).mean() - self._predict(sampled_data_without_feature).mean()

    def _predict(self, x):
        model = self._model
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        if self._method == 'binary_classification_0_logodds':
            outputs = torch.softmax(outputs, dim=1)
            pred_log_odds = torch.log(outputs[:, 0]) - torch.log(outputs[:, 1])
            return pred_log_odds
        elif self._method == 'binary_classification_1_logodds':
            outputs = torch.softmax(outputs, dim=1)
            pred_log_odds = torch.log(outputs[:, 1]) - torch.log(outputs[:, 0])
            return pred_log_odds
        elif self._method == 'binary_classification_0':
            return outputs[:, 0]
        elif self._method == 'binary_classification_1':
            return outputs[:, 1]
        elif self._method == 'regression':
            return outputs
        else:
            logging.error('Uhh this should probably not happen??')
            raise ValueError('Invalid method')
    
    def _all_predictions(self):
        model = self._model
        model.eval()
        with torch.no_grad():
            outputs = model(self._dataset[:][0])
        if self._method == 'binary_classification_0_logodds':
            outputs = torch.softmax(outputs, dim=1)
            pred_log_odds = torch.log(outputs[:, 0]) - torch.log(outputs[:, 1])
            return pred_log_odds
        elif self._method == 'binary_classification_1_logodds':
            outputs = torch.softmax(outputs, dim=1)
            pred_log_odds = torch.log(outputs[:, 1]) - torch.log(outputs[:, 0])
            return pred_log_odds
        if self._method == 'binary_classification_0':
            return outputs[:, 0]
        elif self._method == 'binary_classification_1':
            return outputs[:, 1]
        elif self._method == 'regression':
            return outputs
        else:
            logging.error('Uhh this should probably not happen??')
            raise ValueError('Invalid method')
    
    @property
    def shap_values(self):
        return self._shap_values
    
    @property
    def shap_expectation(self):
        return self._shap_expectation
    
    @property
    def shap_weighted(self):
        return self._shap_weighted