<script lang="ts">
  import { optimizerConfigs, optimizerOptions, lossFunctionOptions } from './optimizer-config';
  import { 
    initializeOptimizerConfig, 
    validateForm, 
    prepareFinalOptimizerConfig, 
    formatScientificNumber, 
    updateArrayValue,
    setTrainingConfig,
    type TrainingConfigRequest
  } from './training-config-utils';
  import {fetchModelDetails, modelDetails} from "../modelDetails"
  import type { TrainConfig, OptimizerConfig } from '../../../../../../source/types/trainTypes';

  import './training-config.css';

  // State variables
  let loading = false;
  let error: string | null = null;
  let result: TrainConfig | null = null;

  // Form fields 
  let epochs: number | null = null;
  let optimizer = '';
  let lossFunction = '';
  let optimizerConfig: OptimizerConfig = { lr: 0.001 };

  let metricsConfig = {
    gradient_visualization: false,
    gradient_visualization_period: 10,
    gradient_norm_visualization: false,
    gradient_norm_visualization_period: 10,
    learning_rate_visualization: false,
    learning_rate_visualization_period: 10,
    weights_visualization: false,
    weights_visualization_period: 10,
    graph_visualization: false,
    profile: false,
    accuracy_visualization: true,
    loss_visualization: true,
    test_validation: true,
    test_validation_period: 10,
    train_validation: true,
    train_validation_period: 1
  };

  let lossFunctionConfig = {
    reduction: 'mean',
    ignore_index: null,
    label_smoothing: 0.0
  };

  // Initialize optimizer config when optimizer changes
  $: if (optimizer) {
    optimizerConfig = initializeOptimizerConfig(optimizer);
  }


  function handleArrayUpdate(key: string, index: number, event: Event) {
    const target = event.target as HTMLInputElement;
    optimizerConfig = updateArrayValue(optimizerConfig, key, index, target.value, optimizer);
  }

  async function handleSubmit() {
    const validationError = validateForm(epochs, optimizer, lossFunction, optimizerConfig);
    if (validationError) {
      error = validationError;
      return;
    }
    
    loading = true;
    error = null;
    result = null;

    const finalOptimizerConfig = prepareFinalOptimizerConfig(optimizer, optimizerConfig);
    
  // In handleSubmit function, update the request object:
  const request: TrainingConfigRequest = {
    epochs: epochs || 10,
    optimizer,
    optimizerConfig: finalOptimizerConfig,
    lossFunction,
    lossFunctionConfig: {
      reduction: lossFunctionConfig.reduction,
      ignore_index: lossFunctionConfig.ignore_index,
      label_smoothing: lossFunctionConfig.label_smoothing
    },
    metrics: metricsConfig
  };

    const response = await setTrainingConfig(request);
    
    if (response.error) {
      error = response.error;
    } else {
      result = response.data || null;
      // Refresh model details after successful configuration
      await fetchModelDetails();
    }
    
    loading = false;
  }
</script>

<div class="container">
  <h1 class="title">Training Configuration</h1>
  
  <div class="section">
    {#if $modelDetails}
      <div class="model-overview">
        <h3 class="model-overview-title">Model Overview</h3>
        <p class="model-overview-text">
          Model Name: <span class="font-semibold">{$modelDetails.name}</span>
        </p>
        {#if $modelDetails.module_graph}
          <p class="model-overview-text">
            Total Layers: <span class="font-semibold">{$modelDetails.module_graph?.layers?.length || 0}</span>
            Total Connections: <span class="font-semibold">{$modelDetails.module_graph?.edges?.length || 0}</span>
          </p>
        {:else}
          <p class="model-overview-warning">
            ⚠️ No module graph found. Please ensure your model is properly configured.
          </p>
        {/if}
        {#if $modelDetails.dataset_config}
          <p class="model-overview-text">
            Dataset: <span class="font-semibold">{$modelDetails.dataset_config.name} (Batch size: {$modelDetails.dataset_config.batch_size})</span>
          </p>
        {:else}
          <p class="model-overview-warning">
            ⚠️ No dataset configuration found. You may want to configure dataset first.
          </p>
        {/if}
      </div>
    {/if}

    <form on:submit|preventDefault={handleSubmit} class="form">
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
                        <span class="array-label">[{arrayIndex}]:</span>
                        <input
                          type="number"
                          value={(optimizerConfig as any)[paramKey]?.[arrayIndex] ?? ''}
                          on:input={(e) => handleArrayUpdate(paramKey, arrayIndex, e)}
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

      <!-- Add this after the optimizer configuration section -->
{#if lossFunction}
  <div class="dynamic-section">
    <h3 class="dynamic-title">Loss Function Configuration</h3>
    <div class="dynamic-input-group">
      <div>
        <label for="reduction" class="input-label">
          Reduction <span class="optional">(optional)</span>
        </label>
        <select
          id="reduction"
          bind:value={lossFunctionConfig.reduction}
          class="input"
          disabled={loading}
        >
          <option value="mean">Mean</option>
          <option value="sum">Sum</option>
          <option value="none">None</option>
        </select>
        <p class="helper-text">Specifies the reduction to apply to the output</p>
      </div>

      <div>
        <label for="ignore_index" class="input-label">
          Ignore Index <span class="optional">(optional)</span>
        </label>
        <input
          id="ignore_index"
          type="number"
          bind:value={lossFunctionConfig.ignore_index}
          placeholder="(e.g., -100)"
          class="input"
          disabled={loading}
        />
        <p class="helper-text">Index to ignore in loss calculation</p>
      </div>

      <div>
        <label for="label_smoothing" class="input-label">
          Label Smoothing <span class="optional">(optional)</span>
        </label>
        <input
          id="label_smoothing"
          type="number"
          bind:value={lossFunctionConfig.label_smoothing}
          placeholder="(e.g., 0.1)"
          min="0"
          max="1"
          step="0.01"
          class="input"
          disabled={loading}
        />
        <p class="helper-text">Label smoothing factor (0.0 to 1.0)</p>
      </div>
    </div>
  </div>
{/if}

<div class="dynamic-section">
  <h3 class="dynamic-title">Training Metrics & Monitoring</h3>
  
  <!-- Visualization Controls -->
  <div class="dynamic-input-group">
    <div class="md:col-span-2">
      <h4 class="text-lg font-medium text-gray-900 mb-3">Visualization Options</h4>
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <div class="checkbox-group">
            <input
              type="checkbox"
              bind:checked={metricsConfig.gradient_visualization}
              class="checkbox"
              disabled={loading}
            />
            <label class="checkbox-label">Gradient Visualization</label>
          </div>
          {#if metricsConfig.gradient_visualization}
            <div class="ml-4">
              <input
                type="number"
                bind:value={metricsConfig.gradient_visualization_period}
                placeholder="Period"
                min="1"
                max="1000"
                class="input w-20"
                disabled={loading}
              />
              <span class="text-sm text-gray-500 ml-2">every N epochs</span>
            </div>
          {/if}
        </div>

        <div class="flex items-center justify-between">
          <div class="checkbox-group">
            <input
              type="checkbox"
              bind:checked={metricsConfig.gradient_norm_visualization}
              class="checkbox"
              disabled={loading}
            />
            <label class="checkbox-label">Gradient Norm Visualization</label>
          </div>
          {#if metricsConfig.gradient_norm_visualization}
            <div class="ml-4">
              <input
                type="number"
                bind:value={metricsConfig.gradient_norm_visualization_period}
                placeholder="Period"
                min="1"
                max="1000"
                class="input w-20"
                disabled={loading}
              />
              <span class="text-sm text-gray-500 ml-2">every N epochs</span>
            </div>
          {/if}
        </div>

        <div class="flex items-center justify-between">
          <div class="checkbox-group">
            <input
              type="checkbox"
              bind:checked={metricsConfig.learning_rate_visualization}
              class="checkbox"
              disabled={loading}
            />
            <label class="checkbox-label">Learning Rate Visualization</label>
          </div>
          {#if metricsConfig.learning_rate_visualization}
            <div class="ml-4">
              <input
                type="number"
                bind:value={metricsConfig.learning_rate_visualization_period}
                placeholder="Period"
                min="1"
                max="1000"
                class="input w-20"
                disabled={loading}
              />
              <span class="text-sm text-gray-500 ml-2">every N epochs</span>
            </div>
          {/if}
        </div>

        <div class="flex items-center justify-between">
          <div class="checkbox-group">
            <input
              type="checkbox"
              bind:checked={metricsConfig.weights_visualization}
              class="checkbox"
              disabled={loading}
            />
            <label class="checkbox-label">Weights Visualization</label>
          </div>
          {#if metricsConfig.weights_visualization}
            <div class="ml-4">
              <input
                type="number"
                bind:value={metricsConfig.weights_visualization_period}
                placeholder="Period"
                min="1"
                max="1000"
                class="input w-20"
                disabled={loading}
              />
              <span class="text-sm text-gray-500 ml-2">every N epochs</span>
            </div>
          {/if}
        </div>
      </div>
    </div>

    <!-- Performance & Monitoring -->
    <div class="md:col-span-2">
      <h4 class="text-lg font-medium text-gray-900 mb-3">Performance & Monitoring</h4>
      <div class="space-y-3">
        <div class="checkbox-group">
          <input
            type="checkbox"
            bind:checked={metricsConfig.graph_visualization}
            class="checkbox"
            disabled={loading}
          />
          <label class="checkbox-label">Graph Visualization</label>
        </div>

        <div class="checkbox-group">
          <input
            type="checkbox"
            bind:checked={metricsConfig.profile}
            class="checkbox"
            disabled={loading}
          />
          <label class="checkbox-label">Enable Profiling</label>
        </div>

        <div class="checkbox-group">
          <input
            type="checkbox"
            bind:checked={metricsConfig.accuracy_visualization}
            class="checkbox"
            disabled={loading}
          />
          <label class="checkbox-label">Accuracy Visualization</label>
        </div>

        <div class="checkbox-group">
          <input
            type="checkbox"
            bind:checked={metricsConfig.loss_visualization}
            class="checkbox"
            disabled={loading}
          />
          <label class="checkbox-label">Loss Visualization</label>
        </div>
      </div>
    </div>

    <!-- Validation Settings -->
    <div class="md:col-span-2">
      <h4 class="text-lg font-medium text-gray-900 mb-3">Validation Settings</h4>
      <div class="space-y-3">
        <div class="flex items-center justify-between">
          <div class="checkbox-group">
            <input
              type="checkbox"
              bind:checked={metricsConfig.test_validation}
              class="checkbox"
              disabled={loading}
            />
            <label class="checkbox-label">Test Validation</label>
          </div>
          <div class="ml-4">
            <input
              type="number"
              bind:value={metricsConfig.test_validation_period}
              placeholder="Period"
              min="1"
              max="1000"
              class="input w-20"
              disabled={loading || !metricsConfig.test_validation}
            />
            <span class="text-sm text-gray-500 ml-2">every N epochs</span>
          </div>
        </div>

        <div class="flex items-center justify-between">
          <div class="checkbox-group">
            <input
              type="checkbox"
              bind:checked={metricsConfig.train_validation}
              class="checkbox"
              disabled={loading}
            />
            <label class="checkbox-label">Train Validation</label>
          </div>
          <div class="ml-4">
            <input
              type="number"
              bind:value={metricsConfig.train_validation_period}
              placeholder="Period"
              min="1"
              max="1000"
              class="input w-20"
              disabled={loading || !metricsConfig.train_validation}
            />
            <span class="text-sm text-gray-500 ml-2">every N epochs</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

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