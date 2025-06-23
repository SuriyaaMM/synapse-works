<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_TRAIN_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';
  import type { TrainConfig, SetTrainConfigArgs, OptimizerConfig, Model } from '../../../../../../source/types';

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

  // Configuration schemas for optimizers
  const optimizerConfigs: Record<string, Record<string, any>> = {
    adadelta: {
      lr: { type: 'number', default: 1.0, min: 0.001, max: 10, step: 0.001, label: 'Learning Rate' },
      rho: { type: 'number', default: 0.9, min: 0, max: 1, step: 0.01, label: 'Rho (ρ) - decay rate' },
      eps: { type: 'number', default: 1e-6, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' }
    },
    adafactor: {
      lr: { type: 'number', default: 1e-3, min: 1e-5, max: 1, step: 1e-5, label: 'Learning Rate', format: 'scientific' },
      eps: { type: 'number', default: 1e-6, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0.0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' },
      beta2_decay: {type: 'number', default: -0.8, min: -1, max: 0, step: 0.01, label: 'Beta2 Decay (for squared gradient averaging)'},
      d: {type: 'number', default: 1.0, min: 0.1, max: 10, step: 0.1,label: 'Clipping Threshold (d)'}
    },
    adam: {
      lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001 }
    },
    adamw: {
      lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0.01, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001 }
    },
    sparseadam: {
      lr: { type: 'number', default: 0.001, min: 1e-5, max: 1, step: 1e-5, label: 'Learning Rate', format: 'scientific' },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001 },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' }
    },
    adamax: {
      lr: { type: 'number', default: 0.002, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001 },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' }
    },
    asgd: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      lambd: { type: 'number', default: 0.0001, min: 0, max: 1, step: 0.0001, label: 'Lambda (λ)' },
      alpha: { type: 'number', default: 0.75, min: 0, max: 1, step: 0.01, label: 'Alpha (α)' },
      t0: { type: 'number', default: 1000000, min: 1, max: 10000000, step: 1, label: 'T0 (averaging start point)' },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' }
    },
    lbfgs: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      max_iter: { type: 'number', default: 20, min: 1, max: 1000, step: 1, label: 'Max Iterations' },
      max_eval: { type: 'number', default: 25, min: 1, max: 1000, step: 1, label: 'Max Function Evaluations' },
      tolerance_grad: { type: 'number', default: 1e-7, min: 1e-12, max: 1e-3, step: 1e-8, label: 'Gradient Tolerance', format: 'scientific' },
      tolerance_change: { type: 'number', default: 1e-9, min: 1e-15, max: 1e-3, step: 1e-10, label: 'Change Tolerance', format: 'scientific' },
      history_size: { type: 'number', default: 100, min: 1, max: 1000, step: 1, label: 'History Size' }
    },
    radam: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001 },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0.0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' }
    },
    rmsprop: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      alpha: { type: 'number', default: 0.99, min: 0, max: 1, step: 0.01, label: 'Alpha (smoothing constant)' },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific' },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' },
      momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum' }
    },
    rprop: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      etas: { type: 'array', default: [0.5, 1.2], label: 'Eta Parameters (η-, η+)', subtype: 'number', min: 0.1, max: 2, step: 0.1 },
      step_sizes: { type: 'array', default: [1e-6, 50], label: 'Step Size Range (min, max)', subtype: 'number', min: 1e-8, max: 100, step: 0.00000001 }
    },
    sgd: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate' },
      momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum' },
      dampening: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Dampening' },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay' },
      nesterov: { type: 'boolean', default: false, label: 'Nesterov Momentum' }
    },
  };

  // Dropdown options for optimizers
  const optimizerOptions = [
    { value: 'adadelta', label: 'Adadelta' },
    { value: 'adafactor', label: 'Adafactor' },
    { value: 'adam', label: 'Adam' },
    { value: 'adamw', label: 'AdamW' },
    { value: 'sparseadam', label: 'SparseAdam' },
    { value: 'adamax', label: 'Adamax' },
    { value: 'asgd', label: 'ASGD (Averaged Stochastic Gradient Descent)' },
    { value: 'lbfgs', label: 'L-BFGS' },
    { value: 'radam', label: 'RAdam' },
    { value: 'rmsprop', label: 'RMSprop' },
    { value: 'rprop', label: 'Rprop' },
    { value: 'sgd', label: 'SGD (Stochastic Gradient Descent)' }
  ];

  const lossFunctionOptions = [
    { value: 'ce', label: 'Cross Entropy' },
    { value: 'bce', label: 'Binary Cross Entropy' }
  ];

  // Initialize optimizer config when optimizer changes
  $: if (optimizer) {
    initializeOptimizerConfig();
  }

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
      if ((optimizerConfig as any)[key] === undefined) {
        (newConfig as any)[key] = Array.isArray(paramConfig.default) 
          ? [...paramConfig.default] 
          : paramConfig.default;
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
        const value = (optimizerConfig as any)[key];
        
        if (paramConfig.type === 'number') {
          if (typeof value !== 'number' || isNaN(value)) {
            return `${paramConfig.label} must be a valid number`;
          }
          if (value < paramConfig.min || value > paramConfig.max) {
            return `${paramConfig.label} must be between ${paramConfig.min} and ${paramConfig.max}`;
          }
        } else if (paramConfig.type === 'array') {
          if (!Array.isArray(value)) {
            return `${paramConfig.label} must be an array`;
          }
          if (paramConfig.subtype === 'number') {
            for (let i = 0; i < value.length; i++) {
              if (typeof value[i] !== 'number' || isNaN(value[i])) {
                return `${paramConfig.label}[${i}] must be a valid number`;
              }
              if (value[i] < paramConfig.min || value[i] > paramConfig.max) {
                return `${paramConfig.label}[${i}] must be between ${paramConfig.min} and ${paramConfig.max}`;
              }
            }
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

  function updateArrayValue(key: string, index: number, value: string) {
    const currentArray = (optimizerConfig as any)[key] as number[] || [];
    const numValue = parseFloat(value);
    if (!isNaN(numValue)) {
      currentArray[index] = numValue;
      optimizerConfig = { ...optimizerConfig, [key]: [...currentArray] };
    }
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

      const res = await client.mutate({
        mutation: SET_TRAIN_CONFIG,
        variables: {
          modelId,
          epochs,
          optimizer,
          optimizerConfig: cleanOptimizerConfig,
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
                    {#if layer.type === 'linear' && 'in_features' in layer && 'out_features' in layer}
                      ({layer.in_features} → {layer.out_features})
                    {/if}
                  </span>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      {/if}
      
      <form on:submit|preventDefault={setTrainingConfig} class="space-y-6 max-w-4xl">
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
                <div class="{paramConfig.type === 'array' ? 'md:col-span-2' : ''}">
                  <label for={paramKey} class="block text-sm font-medium text-gray-700 mb-1">
                    {paramConfig.label}
                    <span class="text-red-500">*</span>
                  </label>
                  
                  {#if paramConfig.type === 'number'}
                    {#if paramConfig.format === 'scientific'}
                      <input
                        id={paramKey}
                        type="text"
                        bind:value={(optimizerConfig as any)[paramKey]}
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm font-mono"
                        disabled={loading}
                        placeholder={formatScientificNumber(paramConfig.default)}
                        on:input={(e) => {
                          const val = parseFloat((e.target as HTMLInputElement).value);
                          if (!isNaN(val)) (optimizerConfig as any)[paramKey] = val;
                        }}
                      />
                    {:else}
                      <input
                        id={paramKey}
                        type="number"
                        bind:value={(optimizerConfig as any)[paramKey]}
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
                        bind:checked={(optimizerConfig as any)[paramKey]}
                        class="w-4 h-4 text-purple-600 bg-gray-100 border-gray-300 rounded focus:ring-purple-500 focus:ring-2"
                        disabled={loading}
                      />
                      <label for={paramKey} class="ml-2 text-sm text-gray-600">
                        Enable {paramConfig.label}
                      </label>
                    </div>
                  {:else if paramConfig.type === 'array'}
                    <div class="space-y-2">
                      {#each ((optimizerConfig as any)[paramKey] || paramConfig.default) as arrayValue, arrayIndex}
                        <div class="flex items-center space-x-2">
                          <span class="text-sm text-gray-600 min-w-[60px]">
                            [{arrayIndex}]:
                          </span>
                          <input
                            type="number"
                            value={arrayValue}
                            step={paramConfig.step}
                            min={paramConfig.min}
                            max={paramConfig.max}
                            class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                            disabled={loading}
                            on:input={(e) => updateArrayValue(paramKey, arrayIndex, (e.target as HTMLInputElement).value)}
                          />
                        </div>
                      {/each}
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
                    {:else if paramConfig.type === 'array'}
                      Each value: {paramConfig.min} - {paramConfig.max}
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
        </div>
      {/if}
    </div>
  {/if}
</div>