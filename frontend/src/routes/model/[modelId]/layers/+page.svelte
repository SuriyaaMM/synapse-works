<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { APPEND_LAYER, DELETE_LAYER } from '$lib/mutations'; 
  import { GET_MODEL, VALIDATE_MODEL } from '$lib/queries';
  import LayerForm from './LayerForm.svelte';
  
  import type { Model, LinearLayerConfig, LayerConfig, LayerConfigInput } from '../../../../../../source/types';

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

<div class="container mx-auto p-1">
  <h1 class="text-3xl font-bold mb-4">Add Layer</h1>

  {#if !modelId}
    <!-- No Model ID Error -->
    <div class="p-1 bg-red-100 border border-red-400 text-red-700 rounded">
      <p>No model ID provided in the URL.</p>
      <p class="mt-2">
        <a href="/create-model" class="text-blue-600 underline">
          Go back to create a model
        </a>
      </p>
    </div>
  {:else}
    <!-- Success Message -->
    {#if result && !validationError}
      <div class="mb-6">
        <h2 class="text-2xl font-semibold mb-3 text-green-700">
          Layer Added Successfully
        </h2>
      </div>
    {/if}

    <!-- Validation Error Message -->
    {#if validationError}
      <div class="mb-6 p-4 bg-yellow-100 border border-yellow-400 text-yellow-800 rounded">
        <h3 class="font-semibold mb-2">Model Validation Warning</h3>
        <p class="mb-3">{validationError}</p>
        <button 
          on:click={deleteAddedLayer}
          disabled={deletingLayer}
          class="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {deletingLayer ? 'Deleting Layer...' : 'Delete Added Layer'}
        </button>
      </div>
    {/if}

    <div class="space-y-2">     
      <!-- Current Model Structure -->
      {#if modelDetails}
        <div class="bg-blue-50 p-3 rounded-md">
          <h3 class="font-semibold text-blue-800 mb-2">Current Model Structure</h3>
          <p class="text-sm text-blue-700">
            Total Layers: {modelDetails.layers_config?.length || 0}
          </p>
          {#if modelDetails.layers_config && modelDetails.layers_config.length > 0}
            <div class="mt-2">
              <p class="text-sm text-blue-700 font-medium">Layer Summary:</p>
              {#each modelDetails.layers_config as layer, index}
                <p class="text-xs text-blue-600 ml-2">
                  {index + 1}. {layer.name || `${layer.type}Layer`} ({layer.type})
                  {#if layer.type === 'linear'}
                    - {(layer as LinearLayerConfig).in_features} → {(layer as LinearLayerConfig).out_features}
                  {/if}
                </p>
              {/each}
            </div>
          {:else}
            <p class="text-sm text-blue-700">
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

      <!-- Error Message -->
      {#if error}
        <div class="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      {/if}
    </div>
  {/if}
</div>