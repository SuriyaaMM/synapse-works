<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_TRAIN_CONFIG } from '$lib/mutations'; // You'll need to add this
  import { GET_MODEL } from '$lib/queries';

  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let result: any = null;
  let modelDetails: any = null;

  // Training configuration form fields
  let epochs = 10;
  let optimizer = 'adam';
  let learningRate = 0.001;
  let lossFunction = 'ce';

  // Available options
  const optimizerOptions = [
    { value: 'adam', label: 'Adam' },
    { value: 'sgd', label: 'SGD' },
    { value: 'rmsprop', label: 'RMSprop' },
    { value: 'adagrad', label: 'Adagrad' }
  ];

  const lossFunctionOptions = [
    { value: 'ce', label: 'Cross Entropy' },
    { value: 'mse', label: 'Mean Squared Error' },
    { value: 'mae', label: 'Mean Absolute Error' },
    { value: 'bce', label: 'Binary Cross Entropy' }
  ];

  // Reactive statement to get modelId from URL params
  $: modelId = $page.url.searchParams.get('modelId');

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
      
      // Pre-populate form if training config already exists
      if (modelDetails?.train_config) {
        const config = modelDetails.train_config;
        epochs = config.epochs || 10;
        optimizer = config.optimizer || 'adam';
        learningRate = config.optimizer_config?.lr || 0.001;
        lossFunction = config.loss_function || 'ce';
      }
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }

  function validateForm(): string | null {
    if (epochs <= 0) return 'Epochs must be a positive number';
    if (learningRate <= 0) return 'Learning rate must be a positive number';
    if (!optimizer.trim()) return 'Optimizer is required';
    if (!lossFunction.trim()) return 'Loss function is required';
    return null;
  }

  async function setTrainingConfig() {
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
      console.log('Setting training config with variables:', {
        modelId,
        epochs,
        optimizer,
        learningRate,
        lossFunction
      });

      const res = await client.mutate({
        mutation: SET_TRAIN_CONFIG,
        variables: { 
          modelId,
          epochs,
          optimizer,
          lr: learningRate,
          loss_function: lossFunction
        }
      });
      
      console.log('Mutation response:', res);
      
      if (!res.data?.setTrainConfig) {
        throw new Error('Failed to set training configuration - no data returned');
      }
      
      result = res.data.setTrainConfig;
      
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
  <h1 class="text-3xl font-bold mb-6">Training Configuration</h1>

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
      <p class="text-gray-700">
        Configuring training for model ID: <span class="font-mono bg-gray-100 px-2 py-1 rounded">{modelId}</span>
      </p>
      
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
            <p class="text-xs text-blue-600 mt-2">
              ℹ️ This model already has training configuration. You can update it below.
            </p>
          {/if}
        </div>

        <!-- Layer Summary -->
        {#if modelDetails.layers_config && modelDetails.layers_config.length > 0}
          <div class="bg-gray-50 p-4 rounded-md">
            <h3 class="font-semibold text-gray-800 mb-3">Layer Architecture</h3>
            <div class="space-y-2">
              {#each modelDetails.layers_config as layer, index}
                <div class="flex items-center justify-between p-2 bg-white rounded border">
                  <span class="text-sm font-medium">
                    Layer {index + 1}: {layer.name || layer.type}
                  </span>
                  <span class="text-xs text-gray-600">
                    {layer.type}
                    {#if layer.type === 'linear'}
                      ({layer.in_features} → {layer.out_features})
                    {/if}
                  </span>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      {/if}
      
      <form on:submit|preventDefault={setTrainingConfig} class="space-y-6 max-w-lg">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label for="epochs" class="block text-sm font-medium text-gray-700 mb-1">
              Epochs <span class="text-red-500">*</span>
            </label>
            <input
              id="epochs"
              type="number"
              bind:value={epochs}
              required
              min="1"
              max="10000"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={loading}
            />
            <p class="text-xs text-gray-500 mt-1">Number of training iterations</p>
          </div>

          <div>
            <label for="learningRate" class="block text-sm font-medium text-gray-700 mb-1">
              Learning Rate <span class="text-red-500">*</span>
            </label>
            <input
              id="learningRate"
              type="number"
              bind:value={learningRate}
              step="0.0001"
              required
              min="0.0001"
              max="1"
              class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
              disabled={loading}
            />
            <p class="text-xs text-gray-500 mt-1">Step size for parameter updates</p>
          </div>
        </div>

        <div>
          <label for="optimizer" class="block text-sm font-medium text-gray-700 mb-1">
            Optimizer <span class="text-red-500">*</span>
          </label>
          <select
            id="optimizer"
            bind:value={optimizer}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
            disabled={loading}
          >
            {#each optimizerOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
          <p class="text-xs text-gray-500 mt-1">Optimization algorithm for training</p>
        </div>

        <div>
          <label for="lossFunction" class="block text-sm font-medium text-gray-700 mb-1">
            Loss Function <span class="text-red-500">*</span>
          </label>
          <select
            id="lossFunction"
            bind:value={lossFunction}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
            disabled={loading}
          >
            {#each lossFunctionOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
          <p class="text-xs text-gray-500 mt-1">Function to measure prediction error</p>
        </div>

        <div class="flex space-x-3 pt-4">
          <button 
            type="submit"
            disabled={loading}
            class="flex-1 px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Saving Configuration...' : 'Save Training Config'}
          </button>
          
          <a 
            href={`/dataset-config?modelId=${modelId}`}
            class="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors whitespace-nowrap"
          >
            Next: Dataset Config
          </a>
          
          <a 
            href={`/append-layer?modelId=${modelId}`}
            class="px-4 py-3 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
          >
            Back to Layers
          </a>
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
            Training Configuration Saved Successfully
          </h2>
          <div class="bg-green-50 p-4 rounded-md">
            <h3 class="font-semibold text-green-800 mb-2">Configuration Summary</h3>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="font-medium text-green-700">Epochs:</span> {result.train_config?.epochs}
              </div>
              <div>
                <span class="font-medium text-green-700">Optimizer:</span> {result.train_config?.optimizer}
              </div>
              <div>
                <span class="font-medium text-green-700">Learning Rate:</span> {result.train_config?.optimizer_config?.lr}
              </div>
              <div>
                <span class="font-medium text-green-700">Loss Function:</span> {result.train_config?.loss_function}
              </div>
            </div>
          </div>
          
          <div class="mt-4 space-x-3">
            <a 
              href={`/dataset-config?modelId=${modelId}`}
              class="inline-block px-6 py-3 bg-gradient-to-r from-green-600 to-green-700 text-white rounded-lg hover:from-green-700 hover:to-green-800 transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              <span class="flex items-center">
                <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                </svg>
                Configure Dataset
              </span>
            </a>
            <button 
              on:click={() => { result = null; error = null; }}
              class="px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Update Configuration
            </button>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>