<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_DATASET_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';

  import type { Model, DatasetConfig, MNISTDatasetConfig, 
                DatasetConfigInput, MNISTDatasetConfigInput, 
                CIFAR10DatasetConfigInput, CIFAR10DatasetConfig } from '../../../../../../source/types';

  import './dataset-config.css';

  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let result: Model | null = null;
  let modelDetails: Model | null = null;

  // Dataset configuration form fields
  let datasetName = 'mnist';
  let batchSize = 32;
  let shuffle = true;
  let trainSplit = 0.7;
  let testSplit = 0.3;

  // MNIST specific variables
  let mnistRoot = './data/mnist';
  let mnistTrain = true;
  let mnistDownload = true;

  // CIFAR-10 specific variables
  let cifar10Root = './data/cifar10';
  let cifar10Train = true;
  let cifar10Download = true;

  // Available dataset options
  const datasetOptions = [
    { value: 'mnist', label: 'MNIST (Handwritten Digits)' },
    { value: 'cifar10', label: 'CIFAR-10 (Colored Images)' },
    { value: 'custom', label: 'Custom Dataset (Coming Soon)', disabled: true }
  ];

  // Extract modelId from URL path instead of query parameters
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    if (modelIndex !== -1 && modelIndex + 1 < pathParts.length) {
      modelId = pathParts[modelIndex + 1];
    } else {
      modelId = null;
    }
  }

  // Reactive statement to ensure splits add up to 1.0
  $: {
    if (trainSplit + testSplit !== 1.0) {
      testSplit = parseFloat((1.0 - trainSplit).toFixed(2));
    }
  }

  // Fetch model details when modelId changes
  $: if (modelId) {
    fetchModelDetails();
  }

  async function fetchModelDetails() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only'
      });
      
      modelDetails = response.data?.getModel;
      
      // Pre-populate form if dataset config already exists
      if (modelDetails?.dataset_config) {
        const config: DatasetConfig = modelDetails.dataset_config;
        datasetName = config.name || 'mnist';
        batchSize = config.batch_size || 32;
        shuffle = config.shuffle !== undefined ? config.shuffle : true;
        
        if (config.split_length && config.split_length.length >= 2) {
          trainSplit = config.split_length[0];
          testSplit = config.split_length[1];
        }
        
        // Handle MNIST config
        const mnistConfig = config as MNISTDatasetConfig;
        if (mnistConfig && datasetName === 'mnist') {
          mnistRoot = mnistConfig.root || './data/mnist';
          mnistTrain = mnistConfig.train !== undefined ? mnistConfig.train : true;
          mnistDownload = mnistConfig.download !== undefined ? mnistConfig.download : true;
        }

        // Handle CIFAR-10 config
        const cifar10Config = config as CIFAR10DatasetConfig;
        if (cifar10Config && datasetName === 'cifar10') {
          cifar10Root = cifar10Config.root || './data/cifar10';
          cifar10Train = cifar10Config.train !== undefined ? cifar10Config.train : true;
          cifar10Download = cifar10Config.download !== undefined ? cifar10Config.download : true;
        }
      }
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }

  function validateForm(): string | null {
    if (batchSize <= 0) return 'Batch size must be a positive number';
    if (trainSplit <= 0 || trainSplit >= 1) return 'Train split must be between 0 and 1';
    if (testSplit <= 0 || testSplit >= 1) return 'Test split must be between 0 and 1';
    if (Math.abs(trainSplit + testSplit - 1.0) > 0.001) return 'Train and test splits must sum to 1.0';
    if (!datasetName.trim()) return 'Dataset name is required';
    if (datasetName === 'mnist' && !mnistRoot.trim()) return 'MNIST root path is required';
    if (datasetName === 'cifar10' && !cifar10Root.trim()) return 'CIFAR-10 root path is required';
    return null;
  }

  async function setDatasetConfig() {
    if (!modelId) {
      error = 'Model ID is missing from URL parameters';
      return;
    }

    const validationError = validateForm();
    if (validationError) {
      error = validationError;
      return;
    }
    
    loading = true;
    error = null;
    result = null;

    try {
      const datasetConfigInput: DatasetConfigInput = {
        name: datasetName,
        batch_size: batchSize,
        shuffle,
        split_length: [trainSplit, testSplit]
      };

      if (datasetName === 'mnist') {
          const mnistConfig: MNISTDatasetConfigInput = {
              root: mnistRoot,
              train: mnistTrain,
              download: mnistDownload
          };
          datasetConfigInput.mnist = mnistConfig;
        }
      else if (datasetName === 'cifar10') {
        const cifar10Config: CIFAR10DatasetConfigInput = {
          root: cifar10Root,
          train: cifar10Train,
          download: cifar10Download
        };
        datasetConfigInput.cifar10 = cifar10Config;
      }

      const res = await client.mutate({
        mutation: SET_DATASET_CONFIG,
        variables: { 
          modelId,
          datasetConfig: datasetConfigInput
        }
      });
      
      console.log('Mutation response:', res);
      
      if (!res.data?.setDataset) {
        throw new Error('Failed to set dataset configuration - no data returned');
      }
      
      result = res.data.setDataset;
      
      // Refresh model details
      await fetchModelDetails();
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }
</script>

<div class="dataset-config-container">
  <h1 class="dataset-config-heading">Dataset Configuration</h1>

  {#if !modelId}
    <div class="dataset-config-error">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="space-y-6">      
      {#if modelDetails}
        <div class="dataset-config-overview">
          <h3 class="font-semibold mb-2">Model Overview</h3>
          <p class="dataset-config-info">
            Model Name: <span>{modelDetails.name}</span>
          </p>
          <p class="dataset-config-info">
            Total Layers: <span>{modelDetails.layers_config?.length || 0}</span>
          </p>
          {#if modelDetails.train_config}
            <p class="dataset-config-info">
              Training: <span>{modelDetails.train_config.epochs} epochs, {modelDetails.train_config.optimizer} optimizer</span>
            </p>
          {:else}
            <p class="dataset-config-warning mt-2">
              ⚠️ No training configuration found. You may want to configure training first.
            </p>
          {/if}
        </div>
      {/if}

      <form on:submit|preventDefault={setDatasetConfig} class="dataset-config-form space-y-6">
        <!-- Dataset Selection -->
        <div>
          <label for="datasetName" class="dataset-config-label">
            Dataset <span class="text-red-500">*</span>
          </label>
          <select
            id="datasetName"
            bind:value={datasetName}
            required
            class="dataset-config-select"
            disabled={loading}
          >
            {#each datasetOptions as option}
              <option value={option.value} disabled={option.disabled || false}>
                {option.label}
              </option>
            {/each}
          </select>
          <p class="dataset-config-description">Choose the dataset for training your model</p>
        </div>

        <!-- General Dataset Settings -->
        <div class="dataset-config-grid">
          <div>
            <label for="batchSize" class="dataset-config-label">
              Batch Size <span class="text-red-500">*</span>
            </label>
            <input
              id="batchSize"
              type="number"
              bind:value={batchSize}
              required
              min="1"
              max="1024"
              class="dataset-config-input"
              disabled={loading}
            />
            <p class="dataset-config-description">Samples per training batch</p>
          </div>

          <div>
            <label for="trainSplit" class="dataset-config-label">
              Train Split <span class="text-red-500">*</span>
            </label>
            <input
              id="trainSplit"
              type="number"
              bind:value={trainSplit}
              step="0.1"
              required
              min="0.1"
              max="0.9"
              class="dataset-config-input"
              disabled={loading}
            />
            <p class="dataset-config-description">Fraction for training</p>
          </div>

          <div>
            <label for="testSplit" class="dataset-config-label">
              Test Split <span class="text-red-500">*</span>
            </label>
            <input
              id="testSplit"
              type="number"
              bind:value={testSplit}
              step="0.1"
              required
              min="0.1"
              max="0.9"
              class="dataset-config-input"
              disabled={loading}
              readonly
            />
            <p class="dataset-config-description">Auto-calculated (1 - train split)</p>
          </div>
        </div>

        <div class="dataset-config-checkbox">
          <input
            id="shuffle"
            type="checkbox"
            bind:checked={shuffle}
            class="h-4 w-4"
            disabled={loading}
          />
          <label for="shuffle" class="ml-2 block text-sm">
            Shuffle dataset
          </label>
          <p class="dataset-config-checkbox-description">Randomize data order during training</p>
        </div>

        <!-- MNIST Specific Settings -->
        {#if datasetName === 'mnist'}
          <div class="dataset-config-section">
            <h3 class="dataset-config-section-title">MNIST Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="mnistRoot" class="dataset-config-label">
                  Data Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="mnistRoot"
                  type="text"
                  bind:value={mnistRoot}
                  required
                  placeholder="./data/mnist"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Local path where MNIST data will be stored</p>
              </div>

              <div class="dataset-config-grid">
                <div class="dataset-config-checkbox">
                  <input
                    id="mnistTrain"
                    type="checkbox"
                    bind:checked={mnistTrain}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="mnistTrain" class="ml-2 block text-sm">
                    Use training set
                  </label>
                </div>

                <div class="dataset-config-checkbox">
                  <input
                    id="mnistDownload"
                    type="checkbox"
                    bind:checked={mnistDownload}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="mnistDownload" class="ml-2 block text-sm">
                    Auto-download if missing
                  </label>
                </div>
              </div>
            </div>
          </div>
        {/if}

        <!-- CIFAR-10 Specific Settings -->
        {#if datasetName === 'cifar10'}
          <div class="dataset-config-section">
            <h3 class="dataset-config-section-title">CIFAR-10 Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="cifar10Root" class="dataset-config-label">
                  Data Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="cifar10Root"
                  type="text"
                  bind:value={cifar10Root}
                  required
                  placeholder="./data/cifar10"
                  class="dataset-config-input"
                  disabled={loading}
                />
                <p class="dataset-config-description">Local path where CIFAR-10 data will be stored</p>
              </div>

              <div class="dataset-config-grid">
                <div class="dataset-config-checkbox">
                  <input
                    id="cifar10Train"
                    type="checkbox"
                    bind:checked={cifar10Train}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="cifar10Train" class="ml-2 block text-sm">
                    Use training set
                  </label>
                </div>

                <div class="dataset-config-checkbox">
                  <input
                    id="cifar10Download"
                    type="checkbox"
                    bind:checked={cifar10Download}
                    class="h-4 w-4"
                    disabled={loading}
                  />
                  <label for="cifar10Download" class="ml-2 block text-sm">
                    Auto-download if missing
                  </label>
                </div>
              </div>
            </div>
          </div>
        {/if}

        <div class="flex space-x-3 pt-6">
          <button 
            type="submit"
            disabled={loading}
            class="dataset-config-button"
          >
            {loading ? 'Saving Dataset Config...' : 'Save Dataset Configuration'}
          </button>
        </div>
      </form>

      {#if error}
        <div class="dataset-config-form-error">
          {error}
        </div>
      {/if}

      {#if result}
        <div class="dataset-config-success">
          Dataset Configuration Saved Successfully
        </div>
      {/if}
    </div>
  {/if}
</div>
