<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { SET_TRAIN_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';
  import type { TrainConfig, OptimizerConfig, Model } from '../../../../../../source/types';

   // State variables
  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let result: TrainConfig | null = null;
  let modelDetails: Model | null = null;

  // Form fields 
  let epochs: number | null = null;
  let optimizer = '';
  let lossFunction = '';
  let optimizerConfig: OptimizerConfig = {lr: 0.001}; // lr is required

  // Configuration schemas for optimizers
  const optimizerConfigs: Record<string, Record<string, any>> = {
    adadelta: {
      lr: { type: 'number', default: 1.0, min: 0.001, max: 10, step: 0.001, label: 'Learning Rate', required: true },
      rho: { type: 'number', default: 0.9, min: 0, max: 1, step: 0.01, label: 'Rho (ρ) - decay rate', required: false },
      eps: { type: 'number', default: 1e-6, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
    },
    adafactor: {
      lr: { type: 'number', default: 1e-2, min: 1e-5, max: 1, step: 1e-5, label: 'Learning Rate', format: 'scientific', required: true },
      eps: { type: 'number', default: 1e-3, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 1e-2, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
      beta2_decay: {type: 'number', default: -0.8, min: -1, max: 0, step: 0.01, label: 'Beta2 Decay (for squared gradient averaging)', required: false},
      d: {type: 'number', default: 1.0, min: 0.1, max: 10, step: 0.1,label: 'Clipping Threshold (d)', required: false}
    },
    adam: {
      lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false }
    },
    adamw: {
      lr: { type: 'number', default: 1e-3, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 0.01, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false }
    },
    sparseadam: {
      lr: { type: 'number', default: 0.001, min: 1e-5, max: 1, step: 1e-5, label: 'Learning Rate', format: 'scientific', required: true },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false }
    },
    adamax: {
      lr: { type: 'number', default: 0.002, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
    },
    asgd: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      lambd: { type: 'number', default: 0.0001, min: 0, max: 1, step: 0.0001, label: 'Lambda (λ)', required: false },
      alpha: { type: 'number', default: 0.75, min: 0, max: 1, step: 0.01, label: 'Alpha (α)', required: false },
      t0: { type: 'number', default: 1000000, min: 1, max: 10000000, step: 1, label: 'T0 (averaging start point)', required: false },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
    },
    lbfgs: {
      lr: { type: 'number', default: 1.0, min: 0.001, max: 10, step: 0.001, label: 'Learning Rate', required: true },
      max_iter: { type: 'number', default: 20, min: 1, max: 1000, step: 1, label: 'Max Iterations', required: false },
      max_eval: { type: 'number', default: 25, min: 1, max: 1000, step: 1, label: 'Max Function Evaluations', required: false },
      tolerance_grad: { type: 'number', default: 1e-7, min: 1e-12, max: 1e-3, step: 1e-8, label: 'Gradient Tolerance', format: 'scientific', required: false },
      tolerance_change: { type: 'number', default: 1e-9, min: 1e-15, max: 1e-3, step: 1e-10, label: 'Change Tolerance', format: 'scientific', required: false },
      history_size: { type: 'number', default: 100, min: 1, max: 1000, step: 1, label: 'History Size', required: false }
    },
    radam: {
      lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      betas: { type: 'array', default: [0.9, 0.999], label: 'Beta Parameters (β1, β2)', subtype: 'number', min: 0, max: 1, step: 0.001, required: false },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 0.0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false }
    },
    rmsprop: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      alpha: { type: 'number', default: 0.99, min: 0, max: 1, step: 0.01, label: 'Alpha (smoothing constant)', required: false },
      eps: { type: 'number', default: 1e-8, min: 1e-12, max: 1e-4, step: 1e-9, label: 'Epsilon', format: 'scientific', required: false },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
      momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum', required: false }
    },
    rprop: {
      lr: { type: 'number', default: 0.01, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      etas: { type: 'array', default: [0.5, 1.2], label: 'Eta Parameters (η-, η+)', subtype: 'number', min: 0.1, max: 2, step: 0.1, required: false },
      step_sizes: { type: 'array', default: [1e-6, 50], label: 'Step Size Range (min, max)', subtype: 'number', min: 1e-8, max: 100, step: 0.00000001, required: false }
    },
    sgd: {
      lr: { type: 'number', default: 0.001, min: 0.0001, max: 1, step: 0.0001, label: 'Learning Rate', required: true },
      momentum: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Momentum', required: false },
      dampening: { type: 'number', default: 0, min: 0, max: 1, step: 0.01, label: 'Dampening', required: false },
      weight_decay: { type: 'number', default: 0, min: 0, max: 1, step: 0.0001, label: 'Weight Decay', required: false },
      nesterov: { type: 'boolean', default: false, label: 'Nesterov Momentum', required: false }
    },
  };

  // Dropdown options for optimizers
  const optimizerOptions = [
    { value: '', label: 'Select an optimizer...' },
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
    { value: '', label: 'Select a loss function...' },
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
    if (!config) {
      optimizerConfig = { lr: 0.001 };
      return;
    }

    // Initialize with learning rate (required) and clear other optional fields
    optimizerConfig = { lr: config.lr?.default ?? 0.001 };
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
      
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = `Failed to fetch model details: ${
        typeof err === 'object' && err !== null && 'message' in err
          ? (err as { message: string }).message
          : String(err)
      }`;
    } finally {
      loading = false;
    }
  }

  function validateForm(): string | null {
    // Only validate required fields
    if (!epochs || epochs <= 0) return 'Epochs must be a positive number';
    if (!optimizer.trim()) return 'Optimizer is required';
    if (!lossFunction.trim()) return 'Loss function is required';
    
    // Validate learning rate (always required)
    const lr = (optimizerConfig as any).lr;
    if (lr === undefined || lr === null || lr === '' || isNaN(Number(lr))) {
      return 'Learning rate is required and must be a valid number';
    }
    
    // Validate optimizer-specific required parameters and any provided optional parameters
    const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
    if (config) {
      for (const [key, paramConfig] of Object.entries(config)) {
        const value = (optimizerConfig as any)[key];
        
        // Skip validation for optional fields that are empty/null/undefined
        if (!paramConfig.required && (value === undefined || value === null || value === '')) {
          continue;
        }
        
        // For required fields or provided optional fields, validate
        const finalValue = value !== undefined && value !== null && value !== '' ? value : paramConfig.default;
        
        if (paramConfig.type === 'number') {
          if (typeof finalValue !== 'number' || isNaN(finalValue)) {
            return `${paramConfig.label} must be a valid number`;
          }
          if (finalValue < paramConfig.min || finalValue > paramConfig.max) {
            return `${paramConfig.label} must be between ${paramConfig.min} and ${paramConfig.max}`;
          }
        } else if (paramConfig.type === 'array') {
          if (!Array.isArray(finalValue)) {
            return `${paramConfig.label} must be an array`;
          }
          if (paramConfig.subtype === 'number') {
            for (let i = 0; i < finalValue.length; i++) {
              if (typeof finalValue[i] !== 'number' || isNaN(finalValue[i])) {
                return `${paramConfig.label}[${i}] must be a valid number`;
              }
              if (finalValue[i] < paramConfig.min || finalValue[i] > paramConfig.max) {
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

  function updateArrayValue(key: string, index: number, value: string | number) {
    const config = optimizerConfigs[optimizer][key];
    const currentArray = (optimizerConfig as any)[key] || [...config.default];
    const numValue = typeof value === 'number' ? value : parseFloat(value);
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
      // Prepare final optimizer config - only include non-empty values or use defaults for required fields
      const finalOptimizerConfig: any = {};
      const config = optimizerConfigs[optimizer as keyof typeof optimizerConfigs];
      
      if (config) {
        Object.entries(config).forEach(([key, paramConfig]) => {
          const value = (optimizerConfig as any)[key];
          
          // For required fields, always include (use default if empty)
          if (paramConfig.required) {
            finalOptimizerConfig[key] = value !== undefined && value !== null && value !== '' ? value : paramConfig.default;
          } 
          // For optional fields, only include if value is provided
          else if (value !== undefined && value !== null && value !== '') {
            finalOptimizerConfig[key] = value;
          }
          // If no value provided for optional field, use default
          else {
            finalOptimizerConfig[key] = paramConfig.default;
          }
        });
      }

      const res = await client.mutate({
        mutation: SET_TRAIN_CONFIG,
        variables: {
          modelId,
          epochs: epochs || 10, // Use default if null
          optimizer,
          optimizerConfig: finalOptimizerConfig,
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
              placeholder="(e.g., 10)"
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
                    {#if paramConfig.required}
                      <span class="text-red-500">*</span>
                    {:else}
                      <span class="text-gray-400 text-xs">(optional)</span>
                    {/if}
                  </label>
                  
                  {#if paramConfig.type === 'number'}
                    {#if paramConfig.format === 'scientific'}
                      <input
                        id={paramKey}
                        type="text"
                        bind:value={(optimizerConfig as any)[paramKey]}
                        placeholder="(e.g., {formatScientificNumber(paramConfig.default)})"
                        required={paramConfig.required}
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm font-mono"
                        disabled={loading}
                      />
                    {:else}
                      <input
                        id={paramKey}
                        type="number"
                        bind:value={(optimizerConfig as any)[paramKey]}
                        placeholder="(e.g., {paramConfig.default})"
                        required={paramConfig.required}
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
                        Enable {paramConfig.label} (default: {paramConfig.default ? 'enabled' : 'disabled'})
                      </label>
                    </div>
                  {:else if paramConfig.type === 'array'}
                    <div class="space-y-2">
                      {#each paramConfig.default as defaultValue, arrayIndex}
                        <div class="flex items-center space-x-2">
                          <span class="text-sm text-gray-600 min-w-[60px]">
                            [{arrayIndex}]:
                          </span>
                          <input
                            type="number"
                            value={(optimizerConfig as any)[paramKey]?.[arrayIndex] ?? ''}
                            on:input={(e: Event) => {
                              const target = e.target as HTMLInputElement;
                              updateArrayValue(paramKey, arrayIndex, target.value);
                            }}
                            placeholder="(e.g., {defaultValue})"
                            step={paramConfig.step}
                            min={paramConfig.min}
                            max={paramConfig.max}
                            class="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
                            disabled={loading}
                          />
                        </div>
                      {/each}
                    </div>
                  {/if}
                  
                  <p class="text-xs text-gray-500 mt-1">
                    {#if paramConfig.type === 'number'}
                      Range: {paramConfig.min} - {paramConfig.max}, Default: {paramConfig.default}
                      {#if paramConfig.format === 'scientific'}
                        (scientific notation)
                      {/if}
                    {:else if paramConfig.type === 'boolean'}
                      Default: {paramConfig.default ? 'Enabled' : 'Disabled'}
                    {:else if paramConfig.type === 'array'}
                      Each value: {paramConfig.min} - {paramConfig.max}, Default: [{paramConfig.default.join(', ')}]
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