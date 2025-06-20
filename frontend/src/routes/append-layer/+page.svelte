<script lang="ts">
  import { page } from '$app/stores';
  import client from '$lib/apolloClient';
  import { APPEND_LINEAR_LAYER } from '$lib/mutations'; // Using your existing mutation
  import { GET_MODEL } from '$lib/queries'; // Updated to match your schema

  let modelId: string | null = null;
  let loading = false;
  let error: string | null = null;
  let result: any = null;
  let modelDetails: any = null;
  let lastLayerOutputFeatures: number | null = null;

  // Form fields
  let layerType = 'linear';
  let layerName = '';
  let inFeatures = '';
  let outFeatures = '';

  // Reactive statement to get modelId from URL params
  $: modelId = $page.url.searchParams.get('modelId');

  // Fetch model details when modelId changes
  $: if (modelId) {
    fetchModelDetails();
  }

  async function fetchModelDetails() {
    if (!modelId) return;
    
    try {
      const response = await client.query({
        query: GET_MODEL,
        variables: { id: modelId },
        fetchPolicy: 'network-only' // Always fetch fresh data
      });
      
      modelDetails = response.data?.getModel;
      
      // Extract the output features from the last layer
      if (modelDetails?.layers_config && modelDetails.layers_config.length > 0) {
        const lastLayer = modelDetails.layers_config[modelDetails.layers_config.length - 1];
        // Handle different layer types - for linear layers, get from the linear config
        if (lastLayer.type === 'linear') {
          lastLayerOutputFeatures = lastLayer.out_features;
        }
        // Add handling for other layer types as needed
        
        // Auto-populate input features if this is not the first layer
        if (lastLayerOutputFeatures && !inFeatures) {
          inFeatures = lastLayerOutputFeatures.toString();
        }
      } else {
        lastLayerOutputFeatures = null;
      }
    } catch (err) {
      console.error('Error fetching model details:', err);
      error = 'Failed to fetch model details';
    }
  }

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

    // Validate that input features match the previous layer's output features
    if (lastLayerOutputFeatures !== null && inFeaturesNum !== lastLayerOutputFeatures) {
      return `Input features (${inFeaturesNum}) must match the previous layer's output features (${lastLayerOutputFeatures})`;
    }
    
    return null;
  }

  async function appendLayer() {
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
      const inFeaturesNum = parseInt(inFeatures, 10);
      const outFeaturesNum = parseInt(outFeatures, 10);
      const layerNameValue = layerName.trim() || `${layerType}Layer`;
      
      console.log('Using mutation with variables:', {
        modelId,
        type: layerType.trim(),
        name: layerNameValue,
        inFeatures: inFeaturesNum,
        outFeatures: outFeaturesNum
      });

      const res = await client.mutate({
        mutation: APPEND_LINEAR_LAYER,
        variables: { 
          modelId,
          type: layerType.trim(),
          inFeatures: inFeaturesNum,
          outFeatures: outFeaturesNum,
          name: layerNameValue
        }
      });
      
      console.log('Mutation response:', res);
      
      if (!res.data?.appendLayer) {
        throw new Error('Failed to append layer - no data returned');
      }
      
      result = res.data.appendLayer;
      
      // Update the last layer output features for the next layer
      lastLayerOutputFeatures = outFeaturesNum;
      
      // Reset form for next layer, but keep input features as the current output features
      layerName = '';
      inFeatures = outFeaturesNum.toString();
      outFeatures = '';
      
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
  <h1 class="text-3xl font-bold mb-6">Append Linear Layer</h1>

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
        Adding layer to model ID: <span class="font-mono bg-gray-100 px-2 py-1 rounded">{modelId}</span>
      </p>
      
      {#if modelDetails}
        <div class="bg-blue-50 p-4 rounded-md">
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
      
      <form on:submit|preventDefault={appendLayer} class="space-y-4 max-w-md">
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
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="dropout">Dropout</option>
          </select>
        </div>

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

        <button 
          type="submit"
          disabled={loading}
          class="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Adding Layer...' : 'Add Layer'}
        </button>
      </form>

      {#if modelDetails && modelDetails.layers_config && modelDetails.layers_config.length > 0}
        <div class="mt-4">
          <a 
            href={`/train-config?modelId=${modelId}`}
            class="inline-block px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <span class="flex items-center">
              <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
              </svg>
              Configure Training Settings
            </span>
          </a>
          <p class="text-sm text-gray-600 mt-2">
            Ready to configure training? Your model has {modelDetails.layers_config.length} layer{modelDetails.layers_config.length !== 1 ? 's' : ''}.
          </p>
        </div>
      {/if}

      {#if error}
        <div class="p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      {/if}

      {#if result}
        <div class="mt-6">
          <h2 class="text-2xl font-semibold mb-3 text-green-700">
            Layer Added Successfully
          </h2>
          <div class="bg-gray-100 p-4 rounded-md overflow-auto">
            <pre class="text-sm">{JSON.stringify(result, null, 2)}</pre>
          </div>
          
          <div class="mt-4 space-x-3">
            <button 
              on:click={() => { result = null; error = null; }}
              class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Add Another Layer
            </button>
            <a 
              href={`/train-config?modelId=${modelId}`}
              class="inline-block px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
            >
              Configure Training
            </a>
            <a 
              href="/create-model"
              class="inline-block px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              Create New Model
            </a>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>