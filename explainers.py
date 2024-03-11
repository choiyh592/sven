import torch
import logging
from tqdm import tqdm

class Explainer():
    def __init__(self, torchmodel, tensordataset, featurevector_size, method='regression', num_to_sample=None, nan=0, tolerance = 0.01, device='cuda'):
        self._model = torchmodel
        self._method = method
        self._featurevector_size = featurevector_size
        assert method in ['binary_classification_0', 'binary_classification_1', 'regression', 'binary_classification_0_logodds', 'binary_classification_1_logodds'], 'Invalid method'

        # Check for NaNs in the dataset
        if torch.isnan(tensordataset.tensors[0]).any():
            # Warn and convert NaNs to 0
            logging.warning(f'NaN values detected in the dataset. Converting to {nan}.')
            tensordataset.tensors[0][torch.isnan(tensordataset.tensors[0])] = nan
            
        self._dataset = tensordataset
        self._num_instances = len(self._dataset)
        self._num_features = int(len(self._dataset[0][0]) / self._featurevector_size)
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
    
    def _generate_coalition_vectors(self, length, featurevector_size, reversed=False):
        if reversed:
            if length == 1:
                return torch.tensor([[1] * featurevector_size, [0] * featurevector_size], dtype=torch.float32, device=self._device)
            else:
                previous_combinations = self._generate_coalition_vectors(length - 1, featurevector_size, reversed=True)
                zero_appended_r = torch.cat((previous_combinations, torch.ones(previous_combinations.shape[0], featurevector_size, dtype=torch.float32, device=self._device)), dim=1)
                one_appended_r = torch.cat((previous_combinations, torch.zeros(previous_combinations.shape[0], featurevector_size, dtype=torch.float32, device=self._device)), dim=1)
                return torch.cat((zero_appended_r, one_appended_r), dim=0)
        else:
            if length == 1:
                return torch.tensor([[0] * featurevector_size, [1] * featurevector_size], dtype=torch.float32, device=self._device)
            else:
                previous_combinations = self._generate_coalition_vectors(length - 1, featurevector_size)
                zero_appended = torch.cat((previous_combinations, torch.zeros(previous_combinations.shape[0], featurevector_size, dtype=torch.float32, device=self._device)), dim=1)
                one_appended = torch.cat((previous_combinations, torch.ones(previous_combinations.shape[0], featurevector_size, dtype=torch.float32, device=self._device)), dim=1)
                return torch.cat((zero_appended, one_appended), dim=0)
    
    def _calculate_shapley_value_for_all_instances(self):
        all_coalition_vectors = self._generate_coalition_vectors(self._num_features, self._featurevector_size)
        all_coalition_vectors_rev = self._generate_coalition_vectors(self._num_features, self._featurevector_size, reversed=True)

        for instance_idx, instance in enumerate(tqdm(self._dataset[:][0], position=0, desc='[INIT] Calculating SHAP values for instances')):
            self._calculate_shapley_value_for_instance(instance, all_coalition_vectors, all_coalition_vectors_rev, instance_idx)

    def _calculate_shapley_value_for_instance(self, instance, all_coalition_vectors, all_coalition_vectors_rev, instance_idx):
        all_coalesced_instances = all_coalition_vectors.clone()
        all_coalesced_instances *= instance
        for feature_idx in range(self._num_features):
            self._shap_values[instance_idx, feature_idx] += self._calculate_shapley_value_for_feature(all_coalesced_instances, all_coalition_vectors, all_coalition_vectors_rev, feature_idx)
    
    def _calculate_shapley_value_for_feature(self, all_coalesced_instances, all_coalition_vectors_orig, all_coalition_vectors_rev, feature_index):
        all_coalition_vectors = all_coalition_vectors_orig.clone()
        all_coalition_vectors[:, feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 0
        feature_attributions = torch.zeros(all_coalesced_instances.shape[0], device=self._device)
        for coalition_vectors_index, (coalesced_instance, coalition_vector_rev) in enumerate(zip(all_coalesced_instances, all_coalition_vectors_rev)):
            feature_attributions[coalition_vectors_index] += self._calculate_sample_feature_attribution(coalesced_instance, coalition_vector_rev, feature_index)

        num_features = self._num_features

        size_of_coalition_vectors = torch.sum(all_coalition_vectors, dim=1) / self._featurevector_size
        log_of_size_factorial = torch.lgamma(size_of_coalition_vectors + 1)

        size_of_inv_coalition_vectors_minus_i = -size_of_coalition_vectors + num_features - 1
        log_of_size_inv_factorial = torch.lgamma(size_of_inv_coalition_vectors_minus_i + 1)

        coeffcients_for_feature_attributions = torch.exp(log_of_size_factorial + log_of_size_inv_factorial - self._log_of_num_feature_permutations)

        shapley_value_for_feature = torch.sum(coeffcients_for_feature_attributions * feature_attributions)

        return shapley_value_for_feature
        
    def _calculate_sample_feature_attribution(self, coalesced_instance, coalition_vector_rev, feature_index):
        if coalesced_instance[feature_index * self._featurevector_size] == 0:
            return 0
        else:
            sampled_data = self._sampleddata[:][0].clone()
            sampled_data *= coalition_vector_rev
            sampled_data += coalesced_instance

            sampled_data_without_feature = self._sampleddata[:][0].clone()
            coalesced_instance_without_feature = coalesced_instance.clone()
            coalesced_instance_without_feature[feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 0
            coalition_vector_rev_without_feature = coalition_vector_rev.clone()
            coalition_vector_rev_without_feature[feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 1

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

if __name__ == "main":
    pass