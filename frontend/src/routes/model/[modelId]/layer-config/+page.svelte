<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { APPEND_LAYER, DELETE_LAYER } from '$lib/mutations'; 
  import { GET_MODEL, VALIDATE_MODEL } from '$lib/queries';
  import LayerForm from './LayerForm.svelte';
  import type { Model,  LayerConfigInput } from '../../../../../../source/types';
  
  import './layer-config.css';

  // State variables
  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let modelDetails: Model | null = null;
  let result: Model | null = null;
  let validationError: string | null = null;
  let addedLayerId: string | null = null;
  let deletingLayer = false;

  // Layer form reference
  let layerFormRef: LayerForm;

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

  // Fetches model details from the server and updates layer information
  async function fetchModelDetails() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only'
      });
      
      modelDetails = response.data?.getModel;
    } catch (err) {
      error = 'Failed to fetch model details';
    }
  }

  // Validates the model after adding a layer
  async function validateModelStructure() {
    if (!modelId || !modelDetails) return;
    
    try {
      // Determine input dimension based on dataset type
      let inputDimension: number[];
      
      // Check if modelDetails has dataset_type information
      const datasetType = modelDetails.dataset_config?.name?.toLowerCase();
      
      switch (datasetType) {
        case 'mnist':
          inputDimension = [1, 28, 28];
          break;
        case 'cifar10':
          inputDimension = [3, 32, 32];
          break;
        default:
          // Default to MNIST if dataset type is not specified or unknown
          inputDimension = [1, 28, 28];
          console.error(`Unknown dataset type: ${datasetType}, defaulting to MNIST dimensions`);
      }
      
      const response = await client.query({
        query: VALIDATE_MODEL,
        variables: { 
          modelId: modelId, 
          in_dimension: inputDimension 
        },
        fetchPolicy: 'network-only'
      });
      
      const validationResult = response.data?.validateModel;
      
      if (validationResult && validationResult.status && validationResult.status.length > 0) {
        // Get the first validation error message
        validationError = validationResult.status[0].message;
        // Get the last added layer ID for potential deletion
        if (modelDetails.layers_config && modelDetails.layers_config.length > 0) {
          const lastLayer = modelDetails.layers_config[modelDetails.layers_config.length - 1];
          addedLayerId = lastLayer.id || null;
        }
      } else {
        validationError = null;
        addedLayerId = null;
      }
    } catch (err: any) {
      console.error('Validation error:', err);
      validationError = 'Failed to validate model structure';
    }
  }

  // Deletes the recently added layer
  async function deleteAddedLayer() {
    if (!modelId || !addedLayerId) return;
    
    deletingLayer = true;
    
    try {
      const response = await client.mutate({
        mutation: DELETE_LAYER,
        variables: { 
          model_id: modelId, 
          layer_id: addedLayerId 
        }
      });
      
      if (response.data?.deleteLayer) {
        result = response.data.deleteLayer;
        validationError = null;
        addedLayerId = null;
        
        // Refresh model details
        await fetchModelDetails();
      }
    } catch (err: any) {
      error = err.message || 'Failed to delete layer';
    } finally {
      deletingLayer = false;
    }
  }

  // Handles layer form submission
  async function handleLayerSubmit(event: CustomEvent<{ layerConfig: LayerConfigInput }>) {
    if (!modelId) {
      error = 'Model ID is missing from URL parameters';
      return;
    }

    const { layerConfig } = event.detail;
    
    loading = true;
    error = null;
    result = null;
    validationError = null;
    addedLayerId = null;

    try {
      // Execute mutation
      const res = await client.mutate({
        mutation: APPEND_LAYER,
        variables: { modelId, layerConfig }
      });
      
      if (!res.data?.appendLayer) {
        throw new Error('Failed to append layer - no data returned');
      }
      
      result = res.data.appendLayer;
      modelDetails = result; // Update model details with the new result
      
      // Reset form for next layer
      if (layerFormRef) {
        layerFormRef.resetForm();
      }

      // Fetch updated model details
      await fetchModelDetails();
      
      // Validate the model structure after adding the layer
      await validateModelStructure();
      
    } catch (err: any) {
      error = err.message || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  // Handles clearing messages from form
  function handleClearMessages() {
    result = null;
    error = null;
    validationError = null;
    addedLayerId = null;
  }
</script>

<div class="layer-config-container">
  <h1 class="layer-config-heading">Layer Configuration</h1>

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

    {#if result && !validationError}
      <div class="layer-config-success">
        Layer Added Successfully
      </div>
    {/if}

    {#if validationError}
      <div class="layer-config-warning">
        <h3>Model Validation Warning</h3>
        <p class="mb-3">{validationError}</p>
        <button 
          on:click={deleteAddedLayer}
          disabled={deletingLayer}
        >
          {deletingLayer ? 'Deleting Layer...' : 'Delete Added Layer'}
        </button>
      </div>
    {/if}

    <div class="space-y-2">     
      {#if modelDetails}
        <div class="layer-config-model">
          <h3 class="layer-config-model-title">Current Model Structure</h3>
          {#if modelDetails.layers_config && modelDetails.layers_config.length > 0}
            <div class="space-y-2">
              {#each modelDetails.layers_config as layer, index}
                <div class="layer-config-layer">
                  <span>Layer {index + 1}: {layer.name || layer.type}</span>
                  <span>
                    {layer.type}
                    {#if layer.type === 'linear' && 'in_features' in layer && 'out_features' in layer}
                      ({layer.in_features} → {layer.out_features})
                    {/if}
                  </span>
                </div>
              {/each}
            </div>
          {:else}
            <p class="layer-config-no-layers">
              No layers added yet - this will be your first layer
            </p>
          {/if}
        </div>
      {/if}
      
      <!-- Layer Configuration Form -->
      <LayerForm 
        bind:this={layerFormRef}
        {loading}
        on:submit={handleLayerSubmit}
        on:clear={handleClearMessages}
      />

      {#if error}
        <div class="layer-config-message">
          {error}
        </div>
      {/if}
    </div>
  {/if}
</div>
