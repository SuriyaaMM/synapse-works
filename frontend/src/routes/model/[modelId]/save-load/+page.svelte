<script lang="ts">
  import client from '$lib/apolloClient';
  import { SAVE_MODEL, LOAD_MODEL } from '$lib/mutations';
  import type { Model } from '../../../../../../source/types';

  // State variables
  let loading = false;
  let error: string | null = null;
  let successMessage: string | null = null;
  let loadedModel: Model | null = null;
  let saveOperation = false;
  let loadOperation = false;

  
  // Saves the current model
  async function saveModel() {
    loading = true;
    saveOperation = true;
    error = null;
    successMessage = null;

    try {
      const response = await client.mutate({
        mutation: SAVE_MODEL
      });

      successMessage = 'Model saved successfully!';

    } catch (err: any) {
      error = err.message || 'Failed to save model';
    } finally {
      loading = false;
      saveOperation = false;
    }
  }

  
   // Loads a model from storage
  async function loadModel() {
    loading = true;
    loadOperation = true;
    error = null;
    successMessage = null;
    loadedModel = null;

    try {
      const response = await client.mutate({
        mutation: LOAD_MODEL
      });
      
      if (response.data?.load) {
        loadedModel = response.data.load;
        successMessage = `Model loaded successfully!`;
      } else {
        throw new Error('Load operation failed - no model data received');
      }
    } catch (err: any) {
      error = err.message || 'Failed to load model';
    } finally {
      loading = false;
      loadOperation = false;
    }
  }
</script>

<div class="container mx-auto p-1">
  <h1 class="text-3xl font-bold mb-4">Model Management</h1>
  
  <div class="space-y-6">
    <!-- Action Buttons -->
    <div class="flex gap-4 flex-wrap">
      <button
        on:click={saveModel}
        disabled={loading}
        class="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {#if saveOperation && loading}
          <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          Saving...
        {:else}
          💾 Save Model
        {/if}
      </button>

      <button
        on:click={loadModel}
        disabled={loading}
        class="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {#if loadOperation && loading}
          <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          Loading...
        {:else}
          📂 Load Model
        {/if}
      </button>
    </div>

    <!-- Success Message -->
    {#if successMessage}
      <div class="p-4 bg-green-100 border border-green-400 text-green-700 rounded-md">
        <div class="flex items-center gap-2">
          <span class="text-green-600">✅</span>
          {successMessage}
        </div>
      </div>
    {/if}

    <!-- Error Message -->
    {#if error}
      <div class="p-4 bg-red-100 border border-red-400 text-red-700 rounded-md">
        <div class="flex items-center gap-2">
          <span class="text-red-600">❌</span>
          {error}
        </div>
      </div>
    {/if}

    <!-- Loaded Model Details -->
    {#if loadedModel}
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h2 class="text-xl font-semibold text-blue-800 mb-3 flex items-center gap-2">
          <span>📋</span>
          Loaded Model Details
        </h2>
        
        <div class="space-y-3">
          <!-- Model Basic Info -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <span class="text-sm font-medium text-blue-700">Model ID:</span>
              <p class="font-mono text-sm bg-blue-100 px-2 py-1 rounded mt-1">
                {loadedModel.id}
              </p>
            </div>
            <div>
              <span class="text-sm font-medium text-blue-700">Model Name:</span>
              <p class="font-semibold text-blue-900 mt-1">
                {loadedModel.name}
              </p>
            </div>
          </div>

          <!-- Quick Actions -->
          <div class="border-t border-blue-200 pt-3 mt-4">
            <span class="text-sm font-medium text-blue-700 block mb-2">Quick Actions:</span>
            <div class="flex gap-2 flex-wrap">
              <a 
                href="/model/{loadedModel.id}/add-layer"
                class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
              >
                Add Layer
              </a>
              <a 
                href="/model/{loadedModel.id}"
                class="px-3 py-1 bg-gray-600 text-white text-sm rounded hover:bg-gray-700"
              >
                View Model
              </a>
            </div>
          </div>
        </div>
      </div>
    {/if}

    <!-- Help Section -->
    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
      <h3 class="font-semibold text-gray-800 mb-2 flex items-center gap-2">
        <span>💡</span>
        Help
      </h3>
      <div class="text-sm text-gray-600 space-y-2">
        <p>
          <strong>Save Model:</strong> Saves the current model configuration to storage. 
          This preserves your model's structure and settings.
        </p>
        <p>
          <strong>Load Model:</strong> Retrieves a previously saved model from storage. 
          The loaded model will display its structure and layer configuration.
        </p>
        <p class="text-xs text-gray-500 mt-3">
          <strong>Note:</strong> Initially, loaded models may have empty layer configurations 
          that will be populated as you add layers to the model.
        </p>
      </div>
    </div>
  </div>
</div>