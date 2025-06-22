<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { APPEND_LINEAR_LAYER } from '$lib/mutations'; 
  import { GET_MODEL } from '$lib/queries';
  
  import type { Model, LinearLayerConfig, LinearLayerConfigInput, LayerConfig, LayerConfigInput } from '../../../../../../source/types';

  // State variables
  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let modelDetails: Model | null = null;
  let result: Model | null = null;
  let lastLayerOutputFeatures: number | null = null;

  // Form fields
  let layerType = 'linear';
  let layerName = '';
  let inFeatures = '';
  let outFeatures = '';

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
      updateLastLayerInfo();
    } catch (err) {
      error = 'Failed to fetch model details';
    }
  }

  
  //Updates the last layer output features and auto-populates input features
  function updateLastLayerInfo() {
    if (modelDetails?.layers_config && modelDetails.layers_config.length > 0) {
      const lastLayer: LayerConfig = modelDetails.layers_config[modelDetails.layers_config.length - 1];
      
      if (lastLayer.type === 'linear') {
        const linearLayer = lastLayer as LinearLayerConfig;
        lastLayerOutputFeatures = linearLayer.out_features;
        
        // Auto-populate input features for the next layer
        if (!inFeatures) {
          inFeatures = lastLayerOutputFeatures.toString();
        }
      }
    } else {
      lastLayerOutputFeatures = null;
    }
  }

  /**
   * Validates the form fields
   * @returns Error message or null if valid
   */
  function validateForm(): string | null {
    if (!layerType.trim()) return 'Layer type is required';
    
    const inFeaturesNum = Number(inFeatures);
    const outFeaturesNum = Number(outFeatures);
    
    if (!inFeatures || isNaN(inFeaturesNum) || inFeaturesNum <= 0) {
      return 'Input features must be a positive number';
    }
    if (!outFeatures || isNaN(outFeaturesNum) || outFeaturesNum <= 0) {
      return 'Output features must be a positive number';
    }

    // Check if input features match previous layer's output
    if (lastLayerOutputFeatures !== null && inFeaturesNum !== lastLayerOutputFeatures) {
      return `Input features (${inFeaturesNum}) must match the previous layer's output features (${lastLayerOutputFeatures})`;
    }
    
    return null;
  }

  // Appends a new layer to the model
  async function appendLayer() {
    if (!modelId) {
      error = 'Model ID is missing from URL parameters';
      return;
    }

    // Validate form before proceeding
    const validationError = validateForm();
    if (validationError) {
      error = validationError;
      return;
    }
    
    loading = true;
    error = null;
    result = null;

    try {
      // Parse form values
      const inFeaturesNum = parseInt(inFeatures, 10);
      const outFeaturesNum = parseInt(outFeatures, 10);
      const layerNameValue = layerName.trim() || `${layerType}Layer`;

      // Create typed layer configuration
      const linearConfig: LinearLayerConfigInput = {
        name: layerNameValue,
        in_features: inFeaturesNum,
        out_features: outFeaturesNum
      };

      const layerConfig: LayerConfigInput = {
        type: layerType.trim(),
        linear: linearConfig
      };

      // Execute mutation
      const res = await client.mutate({
        mutation: APPEND_LINEAR_LAYER,
        variables: { modelId, layerConfig }
      });
      
      if (!res.data?.appendLayer) {
        throw new Error('Failed to append layer - no data returned');
      }
      
      result = res.data.appendLayer;
      
      // Update state for next layer
      lastLayerOutputFeatures = outFeaturesNum;
      resetFormForNextLayer(outFeaturesNum);
      
      // Refresh model details
      await fetchModelDetails();
    } catch (err: any) {
      error = err.message || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  // Resets form fields for adding the next layer
  function resetFormForNextLayer(previousOutputFeatures: number) {
    layerName = '';
    inFeatures = previousOutputFeatures.toString();
    outFeatures = '';
  }

  // Clears result and error messages
  function clearMessages() {
    result = null;
    error = null;
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
    {#if result}
      <div class="mb-6">
        <h2 class="text-2xl font-semibold mb-3 text-green-700">
          Layer Added Successfully
        </h2>
      </div>
    {/if}

    <div class="space-y-2">     
      <!-- Current Model Structure -->
      {#if modelDetails}
        <div class="bg-blue-50 p-3 rounded-md">
          <h3 class="font-semibold text-blue-800 mb-2">Current Model Structure</h3>
          <p class="text-sm text-blue-700">
            Layers: {modelDetails.layers_config?.length || 0}
          </p>
          {#if lastLayerOutputFeatures !== null}
            <p class="text-sm text-blue-700">
              Last layer output features: <span class="font-mono">{lastLayerOutputFeatures}</span>
            </p>
            <p class="text-xs text-blue-600 mt-1">
              ℹ️ Your next layer's input features must be {lastLayerOutputFeatures}
            </p>
          {:else}
            <p class="text-sm text-blue-700">
              This will be the first layer in your model
            </p>
          {/if}
        </div>
      {/if}
      
      <!-- Layer Configuration Form -->
      <form on:submit|preventDefault={appendLayer} class="space-y-2 max-w-md">
        <!-- Layer Type -->
        <div>
          <label for="layerType" class="block text-sm font-medium text-gray-700 mb-1">
            Layer Type <span class="text-red-500">*</span>
          </label>
          <select
            id="layerType"
            bind:value={layerType}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          >
            <option value="linear">Linear</option>
          </select>
        </div>

        <!-- Layer Name -->
        <div>
          <label for="layerName" class="block text-sm font-medium text-gray-700 mb-1">
            Layer Name <span class="text-gray-400">(optional)</span>
          </label>
          <input
            id="layerName"
            type="text"
            bind:value={layerName}
            placeholder="e.g., InputLinear, HiddenLayer1"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
        </div>

        <!-- Input Features -->
        <div>
          <label for="inFeatures" class="block text-sm font-medium text-gray-700 mb-1">
            Input Features <span class="text-red-500">*</span>
            {#if lastLayerOutputFeatures !== null}
              <span class="text-xs text-gray-500">(must be {lastLayerOutputFeatures})</span>
            {/if}
          </label>
          <input
            id="inFeatures"
            type="number"
            bind:value={inFeatures}
            placeholder="e.g., 784"
            required
            min="1"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            class:border-red-500={lastLayerOutputFeatures !== null && inFeatures && Number(inFeatures) !== lastLayerOutputFeatures}
            disabled={loading}
          />
          {#if lastLayerOutputFeatures !== null && inFeatures && Number(inFeatures) !== lastLayerOutputFeatures}
            <p class="text-xs text-red-600 mt-1">
              Input features must match the previous layer's output features ({lastLayerOutputFeatures})
            </p>
          {/if}
        </div>

        <!-- Output Features -->
        <div>
          <label for="outFeatures" class="block text-sm font-medium text-gray-700 mb-1">
            Output Features <span class="text-red-500">*</span>
          </label>
          <input
            id="outFeatures"
            type="number"
            bind:value={outFeatures}
            placeholder="e.g., 64"
            required
            min="1"
            class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={loading}
          />
        </div>

        <!-- Submit Button -->
        <button 
          type="submit"
          disabled={loading}
          class="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Adding Layer...' : 'Add Layer'}
        </button>
      </form>

      <!-- Error Message -->
      {#if error}
        <div class="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      {/if}
    </div>
  {/if}
</div>