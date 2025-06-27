<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { APPEND_LAYER, DELETE_LAYER } from '$lib/mutations';
  import { GET_MODEL, VALIDATE_MODEL } from '$lib/queries';
  import LayerForm from './LayerForm.svelte';
  import type { Model, LayerConfigInput } from '../../../../../../source/types';
  import { tick } from 'svelte';

  import './layer-config.css';

  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let modelDetails: Model | null = null;
  let result: Model | null = null;
  let validationResult: any = null;
  let deletingLayerId: string | null = null;
  let modifyingLayerId: string | null = null;
  let validationTrigger = 0;

  let layerFormRef: LayerForm;

  // Extract modelId from URL
  $: {
    const pathParts = $page.url.pathname.split('/');
    const modelIndex = pathParts.indexOf('model');
    modelId = (modelIndex !== -1 && modelIndex + 1 < pathParts.length) ? pathParts[modelIndex + 1] : null;
  }

  $: {
  // Clear error message when validation state improves
  if (validationResult && !hasValidationErrors() && error && error.includes('Cannot add layer:')) {
    error = null;
  }
}

  // Fetch model when modelId changes
  $: if (modelId) fetchModelDetails();

  async function fetchModelDetails() {
    if (!modelId) return;
    try {
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only'
      });
      modelDetails = response.data?.getModel;
      
      // Always validate after fetching model details
      if (modelDetails) {
        // Use tick to ensure the DOM is updated before validation
        await tick();
        await validateModelStructure();
      }
    } catch (err) {
      console.error('Failed to fetch model details:', err);
      error = 'Failed to fetch model details';
    }
  }

  async function validateModelStructure() {
    if (!modelId || !modelDetails) return;

    try {
      let inputDimension: number[];
      const datasetType = modelDetails.dataset_config?.name?.toLowerCase();
      switch (datasetType) {
        case 'mnist':
          inputDimension = [1, 28, 28];
          break;
        case 'cifar10':
          inputDimension = [3, 32, 32];
          break;
        default:
          inputDimension = [1, 28, 28];
      }

      const response = await client.query({
        query: VALIDATE_MODEL,
        variables: { modelId: modelId, in_dimension: inputDimension },
        fetchPolicy: 'no-cache',
        errorPolicy: 'all'
      });

      if (response.data?.validateModel) {
        validationResult = { ...response.data.validateModel, status: response.data.validateModel.status || [] };
        validationTrigger++;
        await tick();
      } else {
        validationResult = null;
        validationTrigger++;
      }
    } catch (err) {
      console.error('Validation error:', err);
      validationResult = null;
      validationTrigger++;
    }
  }

  // Also trigger validation when modelDetails changes (reactive statement)
  $: if (modelDetails && validationTrigger === 0) {
    validateModelStructure();
  }

  //Check if model has validation errors
  function hasValidationErrors(): boolean {
    if (!validationResult?.status) return false;
    return validationResult.status.some((status: any) => status.message);
  }

  //Get the last layer's validation message
  function getLastLayerValidationMessage(): string | null {
    if (!validationResult?.status || validationResult.status.length === 0) return null;
    
    const lastLayerStatus = validationResult.status[validationResult.status.length - 1];
    return lastLayerStatus?.message || null;
  }

  async function handleLayerSubmit(event: CustomEvent<{ layerConfig: LayerConfigInput }>) {
    if (!modelId) {
      error = 'Model ID is missing';
      return;
    }

    // Check for validation errors before allowing layer addition
    if (hasValidationErrors()) {
      const lastLayerMessage = getLastLayerValidationMessage();
      error = lastLayerMessage 
        ? `Cannot add layer: ${lastLayerMessage}` 
        : 'Cannot add layer: Model has validation errors. Please fix existing layers first.';
      return;
    }

    const { layerConfig } = event.detail;

    loading = true;
    error = null;
    result = null;

    try {
      const res = await client.mutate({
        mutation: APPEND_LAYER,
        variables: { modelId, layerConfig },
        fetchPolicy: 'no-cache'
      });

      if (!res.data?.appendLayer) {
        throw new Error('Failed to append layer - no data returned');
      }

      result = res.data.appendLayer;
      modelDetails = result;

      await validateModelStructure();

      if (layerFormRef) {
        layerFormRef.resetForm();
      }

    } catch (err) {
      console.error('Error adding layer:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Unknown error occurred';
      }
    } finally {
      loading = false;
    }
  }

  async function deleteLayer(layerId: string) {
    if (!modelId || !layerId) return;

    deletingLayerId = layerId;
    error = null;

    try {
      const response = await client.mutate({
        mutation: DELETE_LAYER,
        variables: { model_id: modelId, layer_id: layerId }
      });

      if (response.data?.deleteLayer) {
        result = response.data.deleteLayer;
        modelDetails = result;
        
        await tick();
        await validateModelStructure();
        
        error = null;
      }
    } catch (err) {
      console.error('Error deleting layer:', err);
      if (err instanceof Error) {
        error = err.message;
      } else {
        error = 'Failed to delete layer';
      }
    } finally {
      deletingLayerId = null;
    }
  }

  function modifyLayer(layerId: string) {
    modifyingLayerId = layerId;
    console.log('Modify layer:', layerId);
  }

  function getLayerValidationInfo(layerId: string) {
    fetchModelDetails();
    if (!validationResult?.status) return null;

    const layerStatus = validationResult.status.find((status: any) => status.layer_id === layerId);
    if (!layerStatus) return null;

    return {
      inputDim: layerStatus.in_dimension || null,
      outputDim: layerStatus.out_dimension || null,
      isValid: !layerStatus.message,
      errorMessage: layerStatus.message || null,
      requiredInputDim: layerStatus.required_in_dimension || null,
      layerId: layerStatus.layer_id || null
    };
  }

  function getOverallValidationStatus() {
    if (!validationResult?.status) return { isValid: true, hasErrors: false };

    const hasErrors = validationResult.status.some((status: any) => status.message);
    return {
      isValid: !hasErrors,
      hasErrors: hasErrors,
      errorCount: validationResult.status.filter((status: any) => status.message).length
    };
  }

  function getInputDimension() {
    if (!validationResult?.status || validationResult.status.length === 0) return null;
    return validationResult.status[0]?.in_dimension || null;
  }

  function getOutputDimension() {
    if (!validationResult?.status || validationResult.status.length === 0) return null;
    return validationResult.status[validationResult.status.length - 1]?.out_dimension || null;
  }

  function handleClearMessages() {
    result = null;
    error = null;
  }

  function formatDimension(dim: number[] | null): string {
    if (!dim || !Array.isArray(dim)) return '';
    return `[${dim.join(', ')}]`;
  }
</script>


<div class="layer-config-container">
  {#if !modelId}
    <div class="layer-config-error">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <div class="layer-config-layout">
      <!-- Main Content Area -->
      <div class="layer-config-main">
        <div class="layer-config-header">
          <h1 class="layer-config-heading">Layer Configuration</h1>
        </div>

        {#if result && getOverallValidationStatus().isValid}
          <div class="layer-config-success">
            Layer Added Successfully
          </div>
        {/if}

        {#if getOverallValidationStatus().hasErrors}
          <div class="layer-config-warning">
            <h3>Model Validation Issues ({getOverallValidationStatus().errorCount} errors)</h3>
            <p class="mb-3">Some layers have validation issues. Please fix the layer issues before adding new layers.</p>
          </div>
        {/if}

        <!-- Show blocking message if there are validation errors -->
        {#if hasValidationErrors()}
          <div class="layer-config-error">
            <h3>⚠️ Cannot Add New Layers</h3>
            <p>Please resolve the validation errors in existing layers before adding new ones.</p>
            {#if getLastLayerValidationMessage()}
              <p><strong>Last layer error:</strong> {getLastLayerValidationMessage()}</p>
            {/if}
          </div>
        {/if}

        <!-- Layer Configuration Form - Disable if there are validation errors -->
        <LayerForm 
          bind:this={layerFormRef}
          loading={loading || hasValidationErrors()}
          on:submit={handleLayerSubmit}
          on:clear={handleClearMessages}
        />

        {#if error}
          <div class="layer-config-message">
            {error}
          </div>
        {/if}
      </div>
      <!-- Sidebar with Current Model Structure -->
      <div class="layer-config-sidebar">
        <!-- Input Dimension Display -->
        {#if getInputDimension()}
          <div class="layer-config-dimension-box input-dim">
            <div class="dimension-label">Input Dimension:</div>
            <div class="dimension-value">{formatDimension(getInputDimension())}</div>
          </div>
        {/if}

        <!-- Current Model Layers -->
        {#if modelDetails?.layers_config && modelDetails.layers_config.length > 0}
          <div class="layer-config-layers-list">
            {#each modelDetails.layers_config as layer, index}
              {@const validationInfo = getLayerValidationInfo(layer.id)}
              <div class="layer-config-layer-item" class:invalid={validationInfo && !validationInfo.isValid}>
                <div class="layer-header">
                  <span class="layer-type">{layer.type}: {layer.name || layer.type}</span>
                  <div class="layer-actions">
                    <button 
                      class="layer-action-btn modify"
                      on:click={() => modifyLayer(layer.id)}
                      disabled={modifyingLayerId === layer.id}
                    >
                      Modify
                    </button>
                    <button 
                      class="layer-action-btn delete"
                      on:click={() => deleteLayer(layer.id)}
                      disabled={deletingLayerId === layer.id}
                    >
                      {deletingLayerId === layer.id ? '...' : 'Delete'}
                    </button>
                  </div>
                </div>
                
                <!-- Layer Dimensions - Always show if validation info exists -->
                {#if validationInfo}
                  <!-- Input Dimension -->
                  {#if validationInfo.inputDim && validationInfo.inputDim.length > 0}
                    <div class="layer-dimension input">
                      <strong>Input:</strong> {formatDimension(validationInfo.inputDim)}
                    </div>
                  {/if}
                  
                  <!-- Output Dimension -->
                  {#if validationInfo.outputDim && validationInfo.outputDim.length > 0}
                    <div class="layer-dimension output">
                      <strong>Output:</strong> {formatDimension(validationInfo.outputDim)}
                    </div>
                  {/if}
                  
                  <!-- Validation Status -->
                  {#if validationInfo.isValid}
                    <div class="layer-status valid">
                      ✓ Layer is valid
                    </div>
                  {:else}
                    <div class="layer-status invalid">
                      ✗ Layer has validation issues
                    </div>
                  {/if}
                  
                  <!-- Error Message -->
                  {#if !validationInfo.isValid && validationInfo.errorMessage}
                    <div class="layer-error">
                      <strong>Error:</strong> {validationInfo.errorMessage}
                    </div>
                  {/if}
                  
                  <!-- Required Input Dimension -->
                  {#if validationInfo.requiredInputDim && validationInfo.requiredInputDim.length > 0}
                    <div class="layer-required-dim">
                      <strong>Required Input:</strong> {formatDimension(validationInfo.requiredInputDim)}
                    </div>
                  {/if}
                {:else}
                  <div class="layer-no-validation">
                    No validation data available
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {:else}
          <div class="layer-config-no-layers">
            No layers added yet - this will be your first layer
          </div>
        {/if}

        <!-- Output Dimension Display -->
        {#if getOutputDimension()}
          <div class="layer-config-dimension-box output-dim">
            <div class="dimension-label">Output Dimension:</div>
            <div class="dimension-value">{formatDimension(getOutputDimension())}</div>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>