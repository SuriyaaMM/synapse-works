<script lang="ts">
  import { goto } from '$app/navigation';
  import client from '$lib/apolloClient';
  import { CREATE_MODEL, LOAD_MODEL } from '$lib/mutations';
  import { GET_MODELS } from '$lib/queries';
  import { onMount } from 'svelte';
  
  import type { Model, CreateModelArgs } from '../../../../source/types';

  // Form state
  let modelName: string = '';
  let loading: boolean = false;
  let error: string | null = null;
  let savedModels: Model[] = [];
  let availableModels: Model[] = [];
  let loadingSavedModels: boolean = false;
  let loadingAvailableModels: boolean = false;

  // Create new model and navigate to layer configuration
  async function createModel(): Promise<void> {
    if (!modelName.trim()) {
      error = 'Please enter a model name';
      return;
    }

    loading = true;
    error = null;

    try {
      const createModelArgs: CreateModelArgs = { 
        name: modelName.trim() 
      };

      const res = await client.mutate({
        mutation: CREATE_MODEL,
        variables: createModelArgs
      });

      const model: Model | undefined = res.data?.createModel;
      if (!model?.id) {
        throw new Error('Model creation failed - no ID returned');
      }

      // Reload models after creating a new one
      await loadSavedModels();

      // Navigate to layer configuration page
      await goto(`/model/${model.id}/layers`);
    } catch (err: any) {
      console.error('Apollo Error:', err);
      error = err.message || err.toString() || 'Unknown error occurred';
    } finally {
      loading = false;
    }
  }

  // Load saved models using LOAD_MODEL mutation
  async function loadSavedModels(): Promise<void> {
    loadingSavedModels = true;
    try {
      const res = await client.mutate({
        mutation: LOAD_MODEL
      });
      savedModels = res.data?.load || [];
      
      // If saved models don't have full details, fetch them using GET_MODELS
      if (savedModels.length > 0 && !savedModels[0].layers_config) {
        console.log('Saved models missing details, fetching from GET_MODELS...');
        const detailsRes = await client.query({
          query: GET_MODELS,
          fetchPolicy: 'network-only'
        });
        const modelsWithDetails = detailsRes.data?.getModels || [];
        
        // Match saved models with detailed models by ID
        savedModels = savedModels.map(savedModel => {
          const detailedModel = modelsWithDetails.find((m: Model) => m.id === savedModel.id);
          return detailedModel || savedModel;
        });
      }
    } catch (err) {
      console.error('Error loading saved models:', err);
    } finally {
      loadingSavedModels = false;
    }
  }

  // Load available models using GET_MODELS query
  async function loadAvailableModels(): Promise<void> {
    loadingAvailableModels = true;
    try {
      const res = await client.query({
        query: GET_MODELS,
        fetchPolicy: 'network-only' // Always fetch fresh data
      });
      availableModels = res.data?.getModels || [];
    } catch (err) {
      console.error('Error loading available models:', err);
    } finally {
      loadingAvailableModels = false;
    }
  }

  // Load all models
  async function loadAllModels(): Promise<void> {
    await Promise.all([loadSavedModels(), loadAvailableModels()]);
  }

  // Load models on component mount
  onMount(() => {
    loadAllModels();
  });
</script>

<div class="h-screen flex flex-col p-6">
  <!-- Header -->
  <div class="flex justify-between items-center mb-6">
    <!-- Left-aligned Icon + Title -->
    <div class="flex items-center space-x-3">
      <img src="/brain.png" alt="Brain Icon" class="w-8 h-8" />
      <h1 class="text-3xl font-bold">Synapse Works</h1>
    </div>

    <!-- Right-aligned Profile Icon -->
    <div class="w-10 h-10 rounded-full border border-gray-600 flex items-center justify-center">
      👤
    </div>
  </div>

  <!-- Main content -->
  <div class="flex flex-1 border rounded-lg overflow-hidden shadow-md">
    <!-- Left Panel: Create Model -->
    <div class="w-1/2 p-6 border-r">
      <h2 class="text-2xl font-semibold mb-4">Create Model</h2>

      <form on:submit|preventDefault={createModel} class="space-y-4">
        <input
          type="text"
          bind:value={modelName}
          placeholder="Enter the model name"
          required
          class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />

        <button 
          type="submit" 
          disabled={loading || !modelName.trim()}
          class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Creating...' : 'Create Model'}
        </button>

        {#if error}
          <div class="p-2 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        {/if}
      </form>
    </div>

    <!-- Right Panel: Models -->
    <div class="w-1/2 p-6 flex flex-col">
      <!-- Your Models Section -->
      <div class="mb-8">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xl font-semibold">Your Models</h2>
          <button
            on:click={loadSavedModels}
            disabled={loadingSavedModels}
            class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loadingSavedModels ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {#if loadingSavedModels}
          <div class="flex justify-center items-center py-8">
            <div class="text-gray-500">Loading saved models...</div>
          </div>
        {:else if savedModels.length === 0}
          <p class="text-gray-500">No saved models found. Create your first model to get started!</p>
        {:else}
          <div class="space-y-3 max-h-64 overflow-y-auto">
            {#each savedModels as model}
              <div class="p-4 bg-gray-50 rounded-lg border hover:bg-gray-100 transition-colors">
                <div class="flex justify-between items-start">
                  <div class="flex-1">
                    <h3 class="font-semibold text-lg">{model.name}</h3>
                    <p class="text-sm text-gray-600 mb-2">ID: {model.id}</p>
                    
                    <!-- Layer Configuration Display -->
                    {#if model.layers_config && model.layers_config.length > 0}
                      <div class="mb-2">
                        <p class="text-sm font-medium text-gray-700 mb-1">
                          Layers ({model.layers_config.length}):
                        </p>
                        <div class="space-y-1">
                          {#each model.layers_config as layer}
                            <div class="flex items-center space-x-2 text-xs bg-white px-2 py-1 rounded border">
                              <span class="font-medium text-blue-600">{layer.type}</span>
                              <span class="text-gray-500">•</span>
                              <span class="text-gray-700">{layer.name}</span>
                              <span class="text-gray-400">(ID: {layer.id})</span>
                            </div>
                          {/each}
                        </div>
                      </div>
                    {:else}
                      <p class="text-sm text-gray-600 mb-2">No layers configured</p>
                    {/if}

                    <!-- Dataset Configuration (if available) -->
                    {#if model.dataset_config}
                      <p class="text-sm text-gray-600">
                        Dataset: {model.dataset_config.name}
                      </p>
                    {/if}
                  </div>
                  
                  <div class="flex flex-col gap-2 ml-4">
                    <!-- Navigate to layer configuration -->
                    <button
                      on:click={() => goto(`/model/${model.id}/layers`)}
                      class="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 whitespace-nowrap"
                    >
                      View/Edit
                    </button>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Available Models Section -->
      <div class="border-t pt-6">
        <div class="flex justify-between items-center mb-4">
          <h2 class="text-xl font-semibold">Available Models</h2>
          <button
            on:click={loadAvailableModels}
            disabled={loadingAvailableModels}
            class="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-green-700 disabled:opacity-50"
          >
            {loadingAvailableModels ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        {#if loadingAvailableModels}
          <div class="flex justify-center items-center py-8">
            <div class="text-gray-500">Loading available models...</div>
          </div>
        {:else if availableModels.length === 0}
          <p class="text-gray-500">No available models found.</p>
        {:else}
          <div class="space-y-3 max-h-64 overflow-y-auto">
            {#each availableModels as model}
              <div class="p-4 bg-blue-50 rounded-lg border hover:bg-blue-100 transition-colors">
                <div class="flex justify-between items-start">
                  <div class="flex-1">
                    <h3 class="font-semibold text-lg">{model.name}</h3>
                    <p class="text-sm text-gray-600 mb-2">ID: {model.id}</p>
                    
                    <!-- Layer Configuration Display -->
                    {#if model.layers_config && model.layers_config.length > 0}
                      <div class="mb-2">
                        <p class="text-sm font-medium text-gray-700 mb-1">
                          Layers ({model.layers_config.length}):
                        </p>
                        <div class="space-y-1">
                          {#each model.layers_config as layer}
                            <div class="flex items-center space-x-2 text-xs bg-white px-2 py-1 rounded border">
                              <span class="font-medium text-green-600">{layer.type}</span>
                              <span class="text-gray-500">•</span>
                              <span class="text-gray-700">{layer.name}</span>
                              <span class="text-gray-400">(ID: {layer.id})</span>
                            </div>
                          {/each}
                        </div>
                      </div>
                    {:else}
                      <p class="text-sm text-gray-600 mb-2">No layers configured</p>
                    {/if}

                    <!-- Dataset Configuration (if available) -->
                    {#if model.dataset_config}
                      <p class="text-sm text-gray-600">
                        Dataset: {model.dataset_config.name}
                      </p>
                    {/if}
                  </div>
                  
                  <div class="flex flex-col gap-2 ml-4">
                    <!-- Navigate to layer configuration - same as saved models -->
                    <button
                      on:click={() => goto(`/model/${model.id}/layers`)}
                      class="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700 whitespace-nowrap"
                    >
                      View/Edit
                    </button>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>