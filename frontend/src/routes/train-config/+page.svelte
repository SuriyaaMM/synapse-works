<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_TRAIN_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';
  import type { TrainConfig, SetTrainConfigArgs, OptimizerConfig, Model } from '../../../../source/types';

   // State variables
  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let result: TrainConfig | null = null;
  let modelDetails: Model | null = null;

  // Form fields with defaults
  let epochs = 10;
  let optimizer = 'adam';
  let lossFunction = 'ce';
  let optimizerConfig: OptimizerConfig = { lr: 0.001 };

  // Configuration schemas for different optimizers
  const optimizerConfigs: Record<string, Record<string, any>> = {
    adam: {
      lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' }
    },
    sgd: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum' }
    }
  };

  // Dropdown options
  const optimizerOptions = [
    { value: 'adam', label: 'Adam' },
    { value: 'sgd', label: 'SGD (Stochastic Gradient Descent)' }
  ];

  const lossFunctionOptions = [
    { value: 'ce', label: 'Cross Entropy' },
    { value: 'mse', label: 'Mean Squared Error' }
  ];

  // Initialize optimizer config when optimizer changes
  $: if (optimizer) {
    initializeOptimizerConfig();
  }

  // Reactive statement to get modelId from URL params
  $: modelId = $page.url.searchParams.get('modelId');

  // Fetch model details when modelId changes
  $: if (modelId) {
    fetchModelDetails();
  }

  function initializeOptimizerConfig() {
    const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
    if (!config) return;

    const newConfig: OptimizerConfig = { lr: 0.001 };
    Object.entries(config).forEach(([key, paramConfig]) => {
      // Keep existing value if it exists, otherwise use default
      if (optimizerConfig[key] === undefined) {
        (newConfig as any)[key] = paramConfig.default;
      } else {
        (newConfig as any)[key] = (optimizerConfig as any)[key];
      }
    });
    optimizerConfig = newConfig;
  }

  async function fetchModelDetails() {
    if (!modelId) return;
    
    try {
      loading = true;
      error = null;
      
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only'
      });
      
      if (!response.data?.getModel) {
        throw new Error(`Model with ID ${modelId} not found`);
      }
      
      modelDetails = response.data.getModel;
      
      // Pre-populate form if training config already exists
      if (modelDetails?.train_config) {
        const config = modelDetails.train_config;
        epochs = config.epochs || 10;
        optimizer = config.optimizer || 'adam';
        lossFunction = config.loss_function || 'ce';
        
        // Load optimizer-specific config
        if (config.optimizer_config) {
          optimizerConfig = { ...config.optimizer_config };
          console.log('Loaded optimizer config:', optimizerConfig);
        } else {
          // Initialize with defaults if no existing config
          initializeOptimizerConfig();
        }
      } else {
        // Initialize with defaults for new config
        initializeOptimizerConfig();
      }
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = `Failed to fetch model details: ${
        typeof err === 'object' && err !== null && 'message' in err
          ? (err as { message: string }).message
          : String(err)
      }`;
      
      // Still initialize optimizer config even if model fetch fails
      initializeOptimizerConfig();
    } finally {
      loading = false;
    }
  }

  function validateForm(): string | null {
    if (epochs <= 0) return 'Epochs must be a positive number';
    if (!optimizer.trim()) return 'Optimizer is required';
    if (!lossFunction.trim()) return 'Loss function is required';
    
    // Validate optimizer-specific parameters
    const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
    if (config) {
      for (const [key, paramConfig] of Object.entries(config)) {
        const value = optimizerConfig[key];
        if (paramConfig.type === 'number') {
          if (typeof value !== 'number' || isNaN(value)) {
            return `${paramConfig.label} must be a valid number`;
          }
          if (value < paramConfig.min || value > paramConfig.max) {
            return `${paramConfig.label} must be between ${paramConfig.min} and ${paramConfig.max}`;
          }
        }
      }
    }
    
    return null;
  }

  function formatScientificNumber(value: number): string {
    return value.toExponential(2);
  }

  function parseScientificNumber(value: string): number {
    return parseFloat(value) || 0;
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
    // Clean the optimizer config to remove Apollo Client fields like __typename
    const cleanOptimizerConfig = Object.fromEntries(
      Object.entries(optimizerConfig).filter(([key]) => !key.startsWith('__'))
    );

    const setTrainConfigArgs: SetTrainConfigArgs = {
      model_id: modelId,
      train_config: {
        epochs,
        optimizer,
        optimizer_config: cleanOptimizerConfig,
        loss_function: lossFunction
      }
    };

    const res = await client.mutate({
      mutation: SET_TRAIN_CONFIG,
      variables: {
        modelId,
        epochs,
        optimizer,
        optimizerConfig: cleanOptimizerConfig, // Use cleaned config here too
        loss_function: lossFunction
      }
    });

    console.log('Set training config response:', res);

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
      
      <form on:submit|preventDefault={setTrainingConfig} class="space-y-6 max-w-2xl">
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

        <!-- Dynamic Optimizer Configuration -->
        {#if optimizer && optimizerConfigs[optimizer]}
          <div class="bg-gray-50 p-4 rounded-md">
            <h3 class="font-semibold text-gray-800 mb-3">
              {optimizerOptions.find(opt => opt.value === optimizer)?.label} Configuration
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              {#each Object.entries(optimizerConfigs[optimizer]) as [paramKey, paramConfig]}
                <div>
                  <label for={paramKey} class="block text-sm font-medium text-gray-700 mb-1">
                    {paramConfig.label}
                    <span class="text-red-500">*</span>
                  </label>
                  
                  {#if paramConfig.type === 'number'}
                    {#if paramConfig.format === 'scientific'}
                      <input
                        id={paramKey}
                        type="text"
                        bind:value={optimizerConfig[paramKey]}
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm font-mono"
                        disabled={loading}
                        placeholder={formatScientificNumber(paramConfig.default)}
                        on:input={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) optimizerConfig[paramKey] = val;
                        }}
                      />
                    {:else}
                      <input
                        id={paramKey}
                        type="number"
                        bind:value={optimizerConfig[paramKey]}
                        step={paramConfig.step}
                        min={paramConfig.min}
                        max={paramConfig.max}
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                        disabled={loading}
                      />
                    {/if}
                  {:else if paramConfig.type === 'boolean'}
                    <div class="flex items-center py-2">
                      <input
                        id={paramKey}
                        type="checkbox"
                        bind:checked={optimizerConfig[paramKey]}
                        class="w-4 h-4 text-purple-600 bg-gray-100 border-gray-300 rounded focus:ring-purple-500 focus:ring-2"
                        disabled={loading}
                      />
                      <label for={paramKey} class="ml-2 text-sm text-gray-600">
                        Enable {paramConfig.label}
                      </label>
                    </div>
                  {/if}
                  
                  <p class="text-xs text-gray-500 mt-1">
                    {#if paramConfig.type === 'number'}
                      Range: {paramConfig.min} - {paramConfig.max}
                      {#if paramConfig.format === 'scientific'}
                        (scientific notation)
                      {/if}
                    {:else if paramConfig.type === 'boolean'}
                      Default: {paramConfig.default ? 'Enabled' : 'Disabled'}
                    {/if}
                  </p>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <div class="flex space-x-3 pt-4">
          <button 
            type="submit"
            disabled={loading}
            class="flex-1 px-6 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Saving Configuration...' : 'Save Training Config'}
          </button>
          
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
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <span class="font-medium text-green-700">Epochs:</span> {result.train_config?.epochs}
              </div>
              <div>
                <span class="font-medium text-green-700">Optimizer:</span> {result.train_config?.optimizer}
              </div>
              <div>
                <span class="font-medium text-green-700">Loss Function:</span> {result.train_config?.loss_function}
              </div>
              {#if result.train_config?.optimizer_config}
                {#each Object.entries(result.train_config.optimizer_config) as [key, value]}
                  <div>
                    <span class="font-medium text-green-700 capitalize">{key.replace('_', ' ')}:</span> 
                    {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value}
                  </div>
                {/each}
              {/if}
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
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>