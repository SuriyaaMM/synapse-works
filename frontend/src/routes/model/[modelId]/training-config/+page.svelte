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
    
    const request: TrainingConfigRequest = {
      epochs: epochs || 10,
      optimizer,
      optimizerConfig: finalOptimizerConfig,
      lossFunction
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