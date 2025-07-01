<script lang="ts">
  import client from '$lib/apolloClient';
  import { SET_TRAIN_CONFIG } from '$lib/mutations';
  import { GET_MODEL } from '$lib/queries';
  import type { Model } from '../../../../../../source/types/modelTypes';
  import type { TrainConfig, OptimizerConfig } from '../../../../../../source/types/trainTypes';

  import './training-config.css';

   // State variables
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

  fetchModelDetails();

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
    try {
      loading = true;
      error = null;
      
      const response = await client.query({
        query: GET_MODEL,
        fetchPolicy: 'network-only'
      });
      
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

<div class="container">
  <h1 class="title">Training Configuration</h1>
    <div class="section">
      {#if modelDetails}
        <div class="model-overview">
          <h3 class="model-overview-title">Model Overview</h3>
          <p class="model-overview-text">
            Model Name: <span class="font-semibold">{modelDetails.name}</span>
          </p>
          {#if modelDetails.module_graph}
            <p class="model-overview-text">
              Total Layers: <span class="font-semibold">{modelDetails.module_graph?.layers?.length || 0}</span>
              Totals Connections : <span class="font-semibold">{modelDetails.module_graph?.edges?.length || 0}</span>
            </p>
          {:else}
            <p class="model-overview-warning">
              ⚠️ No module graph found. Please ensure your model is properly configured.
            </p>
          {/if}
          {#if modelDetails.dataset_config}
            <p class="model-overview-text">
              Dataset: <span class="font-semibold">{modelDetails.dataset_config.name} (Batch size :{modelDetails.dataset_config.batch_size}</span>)
            </p>
          {:else}
            <p class="model-overview-warning">
              ⚠️ No dataset configuration found. You may want to configure dataset first.
            </p>
          {/if}
        </div>
      {/if}

      <form on:submit|preventDefault={setTrainingConfig} class="form">
        <div class="input-group">
          <div>
            <label for="epochs" class="input-label">
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
              class="input"
              disabled={loading}
            />
            <p class="helper-text">Number of training iterations</p>
          </div>

          <div>
            <label for="lossFunction" class="input-label">
              Loss Function <span class="required">*</span>
            </label>
            <select
              id="lossFunction"
              bind:value={lossFunction}
              required
              class="input"
              disabled={loading}
            >
              {#each lossFunctionOptions as option}
                <option value={option.value}>{option.label}</option>
              {/each}
            </select>
            <p class="helper-text">Function to measure prediction error</p>
          </div>
        </div>

        <div>
          <label for="optimizer" class="input-label">
            Optimizer <span class="required">*</span>
          </label>
          <select
            id="optimizer"
            bind:value={optimizer}
            required
            class="input"
            disabled={loading}
          >
            {#each optimizerOptions as option}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
          <p class="helper-text">Optimization algorithm for training</p>
        </div>

        {#if optimizer && optimizerConfigs[optimizer]}
          <div class="dynamic-section">
            <h3 class="dynamic-title">
              {optimizerOptions.find(opt => opt.value === optimizer)?.label} Configuration
            </h3>
            <div class="dynamic-input-group">
              {#each Object.entries(optimizerConfigs[optimizer]) as [paramKey, paramConfig]}
                <div class="{paramConfig.type === 'array' ? 'md:col-span-2' : ''}">
                  <label for={paramKey} class="input-label">
                    {paramConfig.label}
                    {#if paramConfig.required}
                      <span class="required">*</span>
                    {:else}
                      <span class="optional">(optional)</span>
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
                        class="input"
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
                        class="input"
                        disabled={loading}
                      />
                    {/if}
                  {:else if paramConfig.type === 'boolean'}
                    <div class="checkbox-group">
                      <input
                        id={paramKey}
                        type="checkbox"
                        bind:checked={(optimizerConfig as any)[paramKey]}
                        class="checkbox"
                        disabled={loading}
                      />
                      <label for={paramKey} class="checkbox-label">
                        Enable {paramConfig.label} (default: {paramConfig.default ? 'enabled' : 'disabled'})
                      </label>
                    </div>
                  {:else if paramConfig.type === 'array'}
                    <div class="section">
                      {#each paramConfig.default as defaultValue, arrayIndex}
                        <div class="array-input">
                          <span class="array-label">[{arrayIndex}] :</span>
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
                            class="input"
                            disabled={loading}
                          />
                        </div>
                      {/each}
                    </div>
                  {/if}

                  <p class="helper-text">
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
            class="submit-button"
          >
            {loading ? 'Saving Configuration...' : 'Save Training Config'}
          </button>
        </div>
      </form>

      {#if error}
        <div class="alert alert-error">
          {error}
        </div>
      {/if}

      {#if result}
        <div class="success-section">
          <h2 class="success-title">
            Training Configuration Saved Successfully
          </h2>
        </div>
      {/if}
    </div>
</div>