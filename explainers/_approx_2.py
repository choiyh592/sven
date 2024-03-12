import torch
import logging
from tqdm import tqdm

class ApproxExplainer2():
    """
    SHAP explainer utilizing the approximation algorithm 2 presented in http://dx.doi.org/10.1007/s10115-013-0679-x by Strumbelj et al.(2014) for PyTorch models.
    """
    def __init__(self, torchmodel, tensordataset, featurevector_size, method='regression', num_to_sample=None, sample_batch_size=None, nan=0, tolerance = 0.01, device='cuda'):
        """
        Initializes the explainer.
        Args:
            torchmodel (torch.nn.Module): The PyTorch model to explain.
            tensordataset (torch.utils.data.TensorDataset): The dataset to explain.
            featurevector_size (int): The size of the feature vector.
            method (str): The method of the model. Must be one of 'binary_classification_0', 'binary_classification_1', 'regression', 'binary_classification_0_logodds', 'binary_classification_1_logodds'.
            num_to_sample (int): The number of instances to sample from the dataset. If None, samples all instances.
            sample_batch_size (int): The size of the batch to sample. If None, samples 1/5 of the dataset.
            nan (int): The value to replace NaNs with.
            tolerance (float): The tolerance for the SHAP values to sum to the expected value.
            device (str): The device to use. Must be one of 'cuda', 'cpu'.
        """
        self._model = torchmodel
        self._method = method
        self._featurevector_size = featurevector_size
        assert method in ['binary_classification_0', 'binary_classification_1', 'regression', 'binary_classification_0_logodds', 'binary_classification_1_logodds'], 'Invalid method'

        if torch.isnan(tensordataset.tensors[0]).any():
            logging.warning(f'NaN values detected in the dataset. Converting to {nan}.')
            tensordataset.tensors[0][torch.isnan(tensordataset.tensors[0])] = nan
            
        self._dataset = tensordataset
        self._num_instances = len(self._dataset)
        self._num_features = int(len(self._dataset[0][0]) / self._featurevector_size)
        
        if sample_batch_size is not None:
            self._sample_batch_size = sample_batch_size
        else:
            self._sample_batch_size = len(self._dataset) // 5
        assert self._sample_batch_size <= len(self._dataset), 'sample_batch_size must be less than or equal to the number of instances in the dataset'

        if num_to_sample is None:
            self._num_batches_to_sample = 5000 // self._sample_batch_size + 1
        else:
            self._num_batches_to_sample = num_to_sample // self._sample_batch_size + 1
                   
        self._sampleddata = torch.cat([self._dataset[torch.randperm(self._num_instances)[:self._sample_batch_size]][:][0] for _ in range(self._num_batches_to_sample)], dim=0)
        
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

        if torch.mean(torch.abs(self._true_predictions - self._shap_summations)) > tolerance:
            logging.warning('SHAP values do not sum to the expected value within the given tolerance.')
    
    def _generate_coalition_vector_indices(self, length):
        random_tensors_list = torch.randint(0, 2, (self._sample_batch_size, length), device='cuda')
        return random_tensors_list
    
    def _calculate_shapley_value_for_all_instances(self):
        all_coalition_vectors = torch.zeros(size=(self._num_batches_to_sample * self._sample_batch_size, self._num_features * self._featurevector_size), device='cuda')
        self._all_coalition_vectors = all_coalition_vectors
        for i in range(self._num_batches_to_sample):
            all_indices = self._generate_coalition_vector_indices(self._num_features)
            for j, indices in enumerate(all_indices):
                for k, idx in enumerate(indices):
                    all_coalition_vectors[j + i * self._sample_batch_size, k * self._featurevector_size : (k + 1) * self._featurevector_size] = idx

        all_coalition_vectors_rev = 1 - all_coalition_vectors

        for instance_idx, instance in enumerate(tqdm(self._dataset[:][0], position=0, desc='[INIT] Calculating SHAP values for instances')):
            self._calculate_shapley_value_for_instance(instance, all_coalition_vectors, all_coalition_vectors_rev, instance_idx)

    def _calculate_shapley_value_for_instance(self, instance, all_coalition_vectors, all_coalition_vectors_rev, instance_idx):
        for feature_idx in range(self._num_features):
            self._shap_values[instance_idx, feature_idx] += self._calculate_shapley_value_for_feature_approx(instance, all_coalition_vectors, all_coalition_vectors_rev, feature_idx)
    
    def _calculate_shapley_value_for_feature_approx(self, instance, all_coalition_vectors_orig, all_coalition_vectors_rev_orig, feature_index):
        all_coalated_instances = all_coalition_vectors_orig.clone()
        all_coalated_instances[:, feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 1
        all_coalated_instances *= instance
        all_coalition_samples = all_coalition_vectors_rev_orig.clone()
        all_coalition_samples[:, feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 0
        all_coalition_samples *= self._sampleddata
        all_coalated_instances += all_coalition_samples

        all_coalated_instances_without_i = all_coalition_vectors_orig.clone()
        all_coalated_instances_without_i[:, feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 0
        all_coalated_instances_without_i *= instance
        all_coalition_samples_with_i = all_coalition_vectors_rev_orig.clone()
        all_coalition_samples_with_i[:, feature_index * self._featurevector_size : (feature_index + 1) * self._featurevector_size] = 1
        all_coalition_samples_with_i *= self._sampleddata
        all_coalated_instances_without_i += all_coalition_samples_with_i

        feature_attributions = self._predict(all_coalated_instances) - self._predict(all_coalated_instances_without_i)
        shapley_value_for_feature = torch.mean(feature_attributions, dim=0).item()

        return shapley_value_for_feature
        
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
        """
        Returns the SHAP values.
        """
        return self._shap_values
    
    @property
    def shap_expectation(self):
        """
        Returns the SHAP expectation.
        """
        return self._shap_expectation

if __name__ == "main":
    pass