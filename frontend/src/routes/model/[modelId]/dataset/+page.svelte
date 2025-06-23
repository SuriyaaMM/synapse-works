<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_DATASET_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';

  import type { Model, DatasetConfig, MNISTDatasetConfig, DatasetConfigInput, MNISTDatasetConfigInput, CIFAR10DatasetConfigInput, CIFAR10DatasetConfig } from '../../../../../../source/types';

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

<div class="container mx-auto p-6">
  <h1 class="text-3xl font-bold mb-6">Dataset Configuration</h1>

  {#if !modelId}
    <div class="p-4 bg-red-100 border border-red-400 text-red-700 rounded">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model" class="text-blue-600 underline">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="space-y-6">      
      {#if modelDetails}
        <div class="bg-blue-50 p-4 rounded-md">
          <h3 class="font-semibold text-blue-800 mb-2">Model Overview</h3>
          <p class="text-sm text-blue-700">
            Model Name: <span class="font-semibold">{modelDetails.name}</span>
          </p>
          <p class="text-sm text-blue-700">
            Total Layers: <span class="font-semibold">{modelDetails.layers_config?.length || 0}</span>
          </p>
          {#if modelDetails.train_config}
            <p class="text-sm text-blue-700">
              Training: <span class="font-semibold">{modelDetails.train_config.epochs} epochs, {modelDetails.train_config.optimizer} optimizer</span>
            </p>
          {:else}
            <p class="text-xs text-orange-600 mt-2">
              ⚠️ No training configuration found. You may want to configure training first.
            </p>
          {/if}
        </div>
      {/if}
      
      <form on:submit|preventDefault={setDatasetConfig} class="space-y-6 max-w-2xl">
        <!-- Dataset Selection -->
        <div>
          <label for="datasetName" class="block text-sm font-medium text-gray-700 mb-1">
            Dataset <span class="text-red-500">*</span>
          </label>
          <select
            id="datasetName"
            bind:value={datasetName}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            disabled={loading}
          >
            {#each datasetOptions as option}
              <option value={option.value} disabled={option.disabled || false}>
                {option.label}
              </option>
            {/each}
          </select>
          <p class="text-xs text-gray-500 mt-1">Choose the dataset for training your model</p>
        </div>

        <!-- General Dataset Settings -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label for="batchSize" class="block text-sm font-medium text-gray-700 mb-1">
              Batch Size <span class="text-red-500">*</span>
            </label>
            <input
              id="batchSize"
              type="number"
              bind:value={batchSize}
              required
              min="1"
              max="1024"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              disabled={loading}
            />
            <p class="text-xs text-gray-500 mt-1">Samples per training batch</p>
          </div>

          <div>
            <label for="trainSplit" class="block text-sm font-medium text-gray-700 mb-1">
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
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              disabled={loading}
            />
            <p class="text-xs text-gray-500 mt-1">Fraction for training</p>
          </div>

          <div>
            <label for="testSplit" class="block text-sm font-medium text-gray-700 mb-1">
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
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              disabled={loading}
              readonly
            />
            <p class="text-xs text-gray-500 mt-1">Auto-calculated (1 - train split)</p>
          </div>
        </div>

        <div class="flex items-center">
          <input
            id="shuffle"
            type="checkbox"
            bind:checked={shuffle}
            class="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
            disabled={loading}
          />
          <label for="shuffle" class="ml-2 block text-sm text-gray-900">
            Shuffle dataset
          </label>
          <p class="ml-2 text-xs text-gray-500">Randomize data order during training</p>
        </div>

        <!-- MNIST Specific Settings -->
        {#if datasetName === 'mnist'}
          <div class="border-t pt-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">MNIST Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="mnistRoot" class="block text-sm font-medium text-gray-700 mb-1">
                  Data Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="mnistRoot"
                  type="text"
                  bind:value={mnistRoot}
                  required
                  placeholder="./data/mnist"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  disabled={loading}
                />
                <p class="text-xs text-gray-500 mt-1">Local path where MNIST data will be stored</p>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="flex items-center">
                  <input
                    id="mnistTrain"
                    type="checkbox"
                    bind:checked={mnistTrain}
                    class="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                    disabled={loading}
                  />
                  <label for="mnistTrain" class="ml-2 block text-sm text-gray-900">
                    Use training set
                  </label>
                </div>

                <div class="flex items-center">
                  <input
                    id="mnistDownload"
                    type="checkbox"
                    bind:checked={mnistDownload}
                    class="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                    disabled={loading}
                  />
                  <label for="mnistDownload" class="ml-2 block text-sm text-gray-900">
                    Auto-download if missing
                  </label>
                </div>
              </div>
            </div>
          </div>
        {/if}

        <!-- CIFAR-10 Specific Settings -->
        {#if datasetName === 'cifar10'}
          <div class="border-t pt-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">CIFAR-10 Dataset Settings</h3>
            
            <div class="space-y-4">
              <div>
                <label for="cifar10Root" class="block text-sm font-medium text-gray-700 mb-1">
                  Data Root Path <span class="text-red-500">*</span>
                </label>
                <input
                  id="cifar10Root"
                  type="text"
                  bind:value={cifar10Root}
                  required
                  placeholder="./data/cifar10"
                  class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
                  disabled={loading}
                />
                <p class="text-xs text-gray-500 mt-1">Local path where CIFAR-10 data will be stored</p>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="flex items-center">
                  <input
                    id="cifar10Train"
                    type="checkbox"
                    bind:checked={cifar10Train}
                    class="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                    disabled={loading}
                  />
                  <label for="cifar10Train" class="ml-2 block text-sm text-gray-900">
                    Use training set
                  </label>
                </div>

                <div class="flex items-center">
                  <input
                    id="cifar10Download"
                    type="checkbox"
                    bind:checked={cifar10Download}
                    class="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                    disabled={loading}
                  />
                  <label for="cifar10Download" class="ml-2 block text-sm text-gray-900">
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
            class="flex-1 px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Saving Dataset Config...' : 'Save Dataset Configuration'}
          </button>
        </div>
      </form>

      {#if error}
        <div class="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      {/if}

      {#if result}
        <div class="mt-6">
          <h2 class="text-2xl font-semibold mb-3 text-green-700">
            Dataset Configuration Saved Successfully
          </h2>
        </div>
      {/if}
    </div>
  {/if}
</div>